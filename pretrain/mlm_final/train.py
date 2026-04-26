import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from dataclasses import asdict

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from .config import ModelConfig
    from .data import StreamingTextDataset, collate_mlm
    from .model import BertForMLM
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import ModelConfig
    from data import StreamingTextDataset, collate_mlm
    from model import BertForMLM


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg, tokenizer):
    ckpt_dir = os.path.join(out_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        os.path.join(ckpt_dir, "checkpoint.pt"),
    )
    with open(os.path.join(ckpt_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    tokenizer.save_pretrained(ckpt_dir)
    latest_file = os.path.join(out_dir, "latest_checkpoint.txt")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(ckpt_dir)


def load_checkpoint(model, optimizer, scheduler, resume_path, device, model_only=False):
    payload = torch.load(resume_path, map_location=device)
    model_state = payload["model"]
    model_pos = model.encoder.embeddings.position_embeddings.weight
    ckpt_pos = model_state.get("encoder.embeddings.position_embeddings.weight")
    if ckpt_pos is not None and ckpt_pos.shape != model_pos.shape:
        if ckpt_pos.shape[1] != model_pos.shape[1]:
            raise ValueError(
                f"Position embedding hidden size mismatch: ckpt={ckpt_pos.shape}, model={model_pos.shape}"
            )
        new_pos = model_pos.detach().clone()
        copy_len = min(ckpt_pos.shape[0], model_pos.shape[0])
        new_pos[:copy_len] = ckpt_pos[:copy_len]
        model_state["encoder.embeddings.position_embeddings.weight"] = new_pos
        print(
            f"Resized position embeddings from {ckpt_pos.shape[0]} to {model_pos.shape[0]} (copied {copy_len})."
        )

    model.load_state_dict(model_state, strict=True)
    if not model_only:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="pretrain/mlm_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--resume_model_only", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=args.seq_len)

    model = BertForMLM(cfg).to(device)

    if device == "cuda":
        print("Optimizing model with torch.compile...")
        model = torch.compile(model)

    dataset = StreamingTextDataset(tokenizer=tokenizer, max_len=args.seq_len, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda b: collate_mlm(b, tokenizer),
    )
    data_iter = iter(loader)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, args.max_steps)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=(device == "cuda" and args.precision == "fp16"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.precision == "fp16"))

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    global_step = 0
    running_loss = 0.0
    last_log_time = time.time()
    tokens_since_log = 0

    resume_ckpt_file = ""
    if args.resume_checkpoint:
        resume_ckpt_file = args.resume_checkpoint
    elif args.resume_latest:
        latest_file = Path(args.out_dir) / "latest_checkpoint.txt"
        if latest_file.exists():
            ckpt_dir = latest_file.read_text(encoding="utf-8").strip()
            resume_ckpt_file = str(Path(ckpt_dir) / "checkpoint.pt")

    if resume_ckpt_file:
        global_step = load_checkpoint(
            model,
            optimizer,
            scheduler,
            resume_ckpt_file,
            device,
            model_only=args.resume_model_only,
        )
        if args.resume_model_only:
            global_step = 0
        print(f"Resumed from: {resume_ckpt_file} at step={global_step}")

    try:
        while global_step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
                    _, loss = model(input_ids, token_type_ids, attention_mask, labels)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                running_loss += loss.item() * args.grad_accum
                tokens_since_log += int(attention_mask.sum().item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                ppl = math.exp(min(avg_loss, 20))
                lr = scheduler.get_last_lr()[0]
                now = time.time()
                dt = max(1e-6, now - last_log_time)
                toks_per_sec = tokens_since_log / dt
                print(f"step={global_step} loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.6e} tok/s={toks_per_sec:.0f}")
                running_loss = 0.0
                tokens_since_log = 0
                last_log_time = now

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, args.out_dir, cfg, tokenizer)
                print(f"Saved checkpoint at step {global_step}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, saving checkpoint...")
    finally:
        if global_step > 0 and (global_step % args.save_every) != 0:
            save_checkpoint(model, optimizer, scheduler, global_step, args.out_dir, cfg, tokenizer)
            print(f"Saved final checkpoint at step {global_step}")


if __name__ == "__main__":
    main()