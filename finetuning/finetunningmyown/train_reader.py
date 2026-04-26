import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .adapters import CustomBertEncoderAdapter
from .reader import ReaderModel, build_feature, load_squad_examples

class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    package_root = Path(__file__).resolve().parents[1]
    package_candidate = package_root / path
    if package_candidate.exists():
        return package_candidate
    return candidate

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = 0.5 * (ce(start_logits, start_pos) + ce(end_logits, end_pos))
            total += float(loss.item())
            count += 1
    return total / max(1, count)

def main():
    parser = argparse.ArgumentParser(description="Train ReaderModel on local SQuAD with custom MLM.")
    parser.add_argument("--train-json", default="squad/train-v1.1.json")
    parser.add_argument("--dev-json", default="squad/dev-v1.1.json")
    parser.add_argument("--checkpoint-dir", default="finetuning/finetunningmyown/artifacts/finetunningmyown")
    parser.add_argument("--encoder-checkpoint", default="pretrain/mlm_final/step_100000")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA is not available! The user requested to train with ONLY GPU."
    device = torch.device("cuda")
    print(f"Device: {device}")

    args.train_json = resolve_path(args.train_json)
    args.dev_json = resolve_path(args.dev_json)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir)
    args.encoder_checkpoint = resolve_path(args.encoder_checkpoint)

    print("Loading tokenizer and encoder...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.encoder_checkpoint), use_fast=True)
    encoder = CustomBertEncoderAdapter(checkpoint_dir=str(args.encoder_checkpoint))

    model = ReaderModel(encoder=encoder).to(device)

    print("Loading SQuAD examples...")
    train_examples = load_squad_examples(str(args.train_json))
    dev_examples = load_squad_examples(str(args.dev_json))

    print(f"Building features for {len(train_examples)} train and {len(dev_examples)} dev examples...")
    train_features = []
    for ex in train_examples:
        feat = build_feature(tokenizer, ex, max_length=args.max_length)
        if feat is not None:
            train_features.append(feat)

    dev_features = []
    for ex in dev_examples:
        feat = build_feature(tokenizer, ex, max_length=args.max_length)
        if feat is not None:
            dev_features.append(feat)
            
    print(f"Kept {len(train_features)} train features and {len(dev_features)} dev features.")

    train_loader = DataLoader(FeatureDataset(train_features), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(FeatureDataset(dev_features), batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(0.1 * total_steps)
    scheduler = make_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    ce = nn.CrossEntropyLoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_dev = float("inf")

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            optimizer.zero_grad(set_to_none=True)
            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_loss = ce(start_logits, start_pos)
            end_loss = ce(end_logits, end_pos)
            loss = 0.5 * (start_loss + end_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running += float(loss.item())

            if global_step % args.log_every == 0:
                avg = running / args.log_every
                print("step {} | train_loss {:.4f}".format(global_step, avg))
                running = 0.0

        print(f"Evaluating epoch {epoch}...")
        dev_loss = evaluate(model, dev_loader, device)
        print("epoch {} done | dev_loss {:.4f}".format(epoch, dev_loss))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "max_length": args.max_length,
        }

        torch.save(checkpoint, ckpt_dir / f"epoch_{epoch}.pt")

        if dev_loss < best_dev:
            best_dev = dev_loss
            print(f"New best dev_loss! Saving to best.pt")
            torch.save(checkpoint, ckpt_dir / "best.pt")

if __name__ == "__main__":
    main()
