import argparse
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .data import (
        BucketBatchSampler,
        TokenizedNMTDataset,
        collate_batch,
        load_and_prepare_dataset,
        tokenize_and_pack,
        train_or_load_sentencepiece,
    )
    from .eval import evaluate, evaluate_bleu, noam_lr_lambda
    from .model_scratch import MODEL_CONFIG, PAD_ID, TransformerScratch
    from .runtime import (
        amp_autocast,
        make_grad_scaler,
        maybe_compile_model,
        resolve_device,
        set_seed,
        setup_file_logging,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from data import (
        BucketBatchSampler,
        TokenizedNMTDataset,
        collate_batch,
        load_and_prepare_dataset,
        tokenize_and_pack,
        train_or_load_sentencepiece,
    )
    from eval import evaluate, evaluate_bleu, noam_lr_lambda
    from model_scratch import MODEL_CONFIG, PAD_ID, TransformerScratch
    from runtime import (
        amp_autocast,
        make_grad_scaler,
        maybe_compile_model,
        resolve_device,
        set_seed,
        setup_file_logging,
    )


UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def build_parser(default_source_lang, default_target_lang, default_cache_dir, default_sp_model_name):
    parser = argparse.ArgumentParser(
        description="Train a scratch-built Transformer for EN<->KN translation (no nn.Transformer modules)."
    )
    parser.add_argument("--source-lang", default=default_source_lang, choices=["en", "kn"])
    parser.add_argument("--target-lang", default=default_target_lang, choices=["en", "kn"])

    parser.add_argument("--dataset-name", default="ai4bharat/samanantar")
    parser.add_argument("--dataset-config", default="kn")
    parser.add_argument("--cache-dir", default=default_cache_dir)
    parser.add_argument("--use-saved-dataset", action="store_true")
    parser.add_argument("--save-dataset-to-disk", action="store_true")
    parser.add_argument("--num-proc", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=600000)
    parser.add_argument("--val-fraction", type=float, default=0.001)
    parser.add_argument("--min-char-len", type=int, default=2)
    parser.add_argument("--max-char-len", type=int, default=500)

    parser.add_argument("--sp-model-name", default=default_sp_model_name)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--tokenizer-pairs", type=int, default=2500000)
    parser.add_argument("--max-seq-len", type=int, default=128)

    parser.add_argument("--d-model", type=int, default=MODEL_CONFIG["d_model"])
    parser.add_argument("--nhead", type=int, default=MODEL_CONFIG["nhead"])
    parser.add_argument("--num-enc-layers", type=int, default=MODEL_CONFIG["num_enc_layers"])
    parser.add_argument("--num-dec-layers", type=int, default=MODEL_CONFIG["num_dec_layers"])
    parser.add_argument("--d-ff", type=int, default=MODEL_CONFIG["d_ff"])
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["dropout"])
    parser.add_argument("--min-params", type=int, default=0, help="Fail if model has fewer params than this threshold.")

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=12000)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bleu-samples", type=int, default=500)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-only", action="store_true", help="Fail immediately if CUDA is unavailable/unsupported.")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop training after this many train batches (0 = full epoch).")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Save step checkpoint every N train batches (0 = disabled).")
    parser.add_argument("--run-name", default="", help="Optional subfolder name under checkpoints for this run.")
    parser.add_argument("--resume-checkpoint", default="", help="Path to checkpoint to resume training from.")
    parser.add_argument("--log-file", default="", help="Path to training log file. Defaults to <ckpt_dir>/train.log.")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for stability.")

    return parser


def run_training(args):
    if args.source_lang == args.target_lang:
        raise ValueError("source-lang and target-lang must be different")

    set_seed(args.seed)
    device = resolve_device(args.device, gpu_only=args.gpu_only)
    use_amp = device.type == "cuda"

    cache_dir = Path(args.cache_dir)
    ckpt_dir = cache_dir / "checkpoints"
    if args.run_name:
        ckpt_dir = ckpt_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file if args.log_file else str(ckpt_dir / "train.log")
    log_path = setup_file_logging(log_file)
    print("Run name:", args.run_name if args.run_name else "default")
    print("Source/Target:", "{}->{}".format(args.source_lang, args.target_lang))

    train_text_ds, val_text_ds = load_and_prepare_dataset(args)
    sp_model_file = train_or_load_sentencepiece(
        train_ds=train_text_ds,
        args=args,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    sp = spm.SentencePieceProcessor(model_file=sp_model_file)
    print("Tokenizer vocab size:", sp.get_piece_size())

    train_tok_ds = tokenize_and_pack(
        dataset=train_text_ds,
        sp=sp,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    val_tok_ds = tokenize_and_pack(
        dataset=val_text_ds,
        sp=sp,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    train_ds = TokenizedNMTDataset(train_tok_ds)
    val_ds = TokenizedNMTDataset(val_tok_ds)

    train_lengths = np.maximum(np.array(train_tok_ds["src_len"]), np.array(train_tok_ds["tgt_len"]))
    val_lengths = np.maximum(np.array(val_tok_ds["src_len"]), np.array(val_tok_ds["tgt_len"]))

    num_workers = max(1, min(4, os.cpu_count() or 1))
    collate_fn = partial(collate_batch, pad_id=PAD_ID)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_lengths, max_tokens=args.max_tokens_per_batch, shuffle=True),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(val_lengths, max_tokens=args.max_tokens_per_batch, shuffle=False),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    cfg = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_enc_layers": args.num_enc_layers,
        "num_dec_layers": args.num_dec_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
    }
    model = TransformerScratch(vocab_size=args.vocab_size, pad_id=PAD_ID, **cfg).to(device)

    resume_ckpt = None
    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError("Resume checkpoint not found: {}".format(resume_path))

        try:
            resume_ckpt = torch.load(str(resume_path), map_location=device, weights_only=True)
        except TypeError:
            resume_ckpt = torch.load(str(resume_path), map_location=device)

        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val_loss = float(resume_ckpt.get("val_loss", best_val_loss))

        hist_path = ckpt_dir / "train_history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as handle:
                history = json.load(handle)

        print("Resuming from checkpoint:", resume_path)
        print("Resume start epoch:", start_epoch)

    model = maybe_compile_model(model, disable_compile=args.disable_compile)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    scaler = make_grad_scaler(use_amp)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lr_lambda(step, args.warmup_steps),
    )

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    param_count = sum(p.numel() for p in model.parameters())
    if args.min_params > 0 and param_count < args.min_params:
        raise RuntimeError(
            "Model has {:,} params, below required minimum {:,}.".format(param_count, args.min_params)
        )

    run_config_path = ckpt_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "model_config": cfg,
                "param_count": param_count,
                "log_file": str(log_path),
            },
            handle,
            ensure_ascii=True,
            indent=2,
        )

    print("Scratch Transformer | params: {:,}".format(param_count))
    print("Train batches: {:,} | Val batches: {:,}".format(len(train_loader), len(val_loader)))

    stop_training = False

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0
        started = time.time()

        for step, (src, tgt) in enumerate(train_loader, start=1):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with amp_autocast(device, use_amp):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            n_tokens = int(tgt_out.ne(PAD_ID).sum().item())
            total_loss += float(loss.item()) * args.grad_accum_steps * n_tokens
            total_tokens += n_tokens

            if step % 200 == 0:
                avg_loss = total_loss / max(1, total_tokens)
                elapsed = time.time() - started
                tok_per_sec = total_tokens / max(1e-6, elapsed)
                print(
                    "Epoch {} | step {:,}/{:,} | loss {:.4f} | ppl {:.2f} | tok/s {:,.0f}".format(
                        epoch,
                        step,
                        len(train_loader),
                        avg_loss,
                        math.exp(min(avg_loss, 20)),
                        tok_per_sec,
                    )
                )

            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                step_path = ckpt_dir / "step_e{:02d}_s{:07d}.pt".format(epoch, step)
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": base_model.state_dict(),
                        "model_config": cfg,
                        "vocab_size": args.vocab_size,
                        "max_seq_len": args.max_seq_len,
                        "sp_model_file": str(sp_model_file),
                        "special_ids": {
                            "pad": PAD_ID,
                            "unk": UNK_ID,
                            "bos": BOS_ID,
                            "eos": EOS_ID,
                        },
                        "source_lang": args.source_lang,
                        "target_lang": args.target_lang,
                    },
                    step_path,
                )
                print("Saved step checkpoint:", step_path)

            if args.max_train_steps > 0 and step >= args.max_train_steps:
                print("Reached max train steps ({}); stopping early.".format(args.max_train_steps))
                stop_training = True
                break

        train_loss = total_loss / max(1, total_tokens)
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            pad_id=PAD_ID,
            amp_autocast_fn=amp_autocast,
        )
        bleu = evaluate_bleu(
            model=model,
            sp=sp,
            val_text_ds=val_text_ds,
            device=device,
            max_seq_len=args.max_seq_len,
            max_samples=args.bleu_samples,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
        )

        elapsed = time.time() - started
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu": bleu,
            "time_sec": elapsed,
        }
        history.append(summary)
        print(
            "Epoch {}/{} done | train {:.4f} (ppl {:.2f}) | val {:.4f} (ppl {:.2f}) | bleu {:.2f} | {:.0f}s".format(
                epoch,
                args.epochs,
                train_loss,
                train_ppl,
                val_loss,
                val_ppl,
                bleu,
                elapsed,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "best.pt"
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "bleu": bleu,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "model_config": cfg,
                    "vocab_size": args.vocab_size,
                    "max_seq_len": args.max_seq_len,
                    "sp_model_file": str(sp_model_file),
                    "special_ids": {
                        "pad": PAD_ID,
                        "unk": UNK_ID,
                        "bos": BOS_ID,
                        "eos": EOS_ID,
                    },
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                },
                best_path,
            )
            print("Saved new best checkpoint:", best_path)

        if stop_training:
            break

    hist_path = ckpt_dir / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=True, indent=2)
    print("Training finished. Best val loss: {:.4f}".format(best_val_loss))
    print("History:", hist_path)


def main(
    default_source_lang="en",
    default_target_lang="kn",
    default_cache_dir="",
    default_sp_model_name="",
):
    if not default_cache_dir:
        default_cache_dir = "translation/scratch_transformer/artifacts/{}_to_{}_scratch".format(
            default_source_lang, default_target_lang
        )
    if not default_sp_model_name:
        default_sp_model_name = "{}_{}_scratch_unigram_v1".format(default_source_lang, default_target_lang)

    parser = build_parser(
        default_source_lang=default_source_lang,
        default_target_lang=default_target_lang,
        default_cache_dir=default_cache_dir,
        default_sp_model_name=default_sp_model_name,
    )
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
