import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from .model import MODEL_CONFIG, PAD_ID, TransformerNMT
except ImportError:  # pragma: no cover - direct script execution fallback
    from model import MODEL_CONFIG, PAD_ID, TransformerNMT


UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def make_grad_scaler(use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(device, use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device, gpu_only=False):
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda" and gpu_only:
        raise RuntimeError("GPU-only mode is enabled but CUDA device was not requested/found.")

    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        arch = "sm_{}{}".format(cap[0], cap[1])
        supported_arches = set(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else set()
        if supported_arches and arch not in supported_arches:
            print(
                "Warning: GPU arch {} not listed in this torch build ({}). "
                "Trying runtime CUDA probe...".format(arch, ", ".join(sorted(supported_arches)))
            )

        try:
            _ = torch.tensor([0.0], device=device)
        except Exception as err:
            if gpu_only:
                raise RuntimeError("CUDA is unavailable in this runtime: {}".format(err))
            print("CUDA is unavailable in this runtime, falling back to CPU: {}".format(err))
            device = torch.device("cpu")

    print("Device:", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM: {:.1f} GB".format(props.total_memory / 1e9))
    return device


def extract_pair(example):
    translation = example.get("translation")
    if isinstance(translation, dict):
        en = translation.get("en")
        kn = translation.get("kn")
        if en and kn:
            return str(en).strip(), str(kn).strip()

    for key in ("src", "source", "en"):
        if key in example:
            src = example.get(key)
            break
    else:
        src = None

    for key in ("tgt", "target", "kn"):
        if key in example:
            tgt = example.get(key)
            break
    else:
        tgt = None

    if src is None or tgt is None:
        return "", ""

    return str(src).strip(), str(tgt).strip()


def load_and_prepare_dataset(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_name_tag = args.dataset_name.replace("/", "__")
    cfg_tag = str(args.dataset_config).replace("/", "__")
    local_ds_dir = cache_dir / "{}_{}".format(ds_name_tag, cfg_tag)

    if local_ds_dir.exists() and args.use_saved_dataset:
        print("Loading dataset from disk:", local_ds_dir)
        ds = load_from_disk(str(local_ds_dir))
    else:
        print("Downloading/loading dataset with Hugging Face datasets...")
        try:
            ds = load_dataset(args.dataset_name, args.dataset_config, cache_dir=str(cache_dir))
        except ValueError as err:
            if args.dataset_name == "ai4bharat/samanantar" and args.dataset_config == "en-kn":
                print("Config 'en-kn' unavailable on this datasets version; retrying with 'kn'.")
                ds = load_dataset(args.dataset_name, "kn", cache_dir=str(cache_dir))
            else:
                raise err
        if args.save_dataset_to_disk:
            print("Saving local dataset copy to:", local_ds_dir)
            ds.save_to_disk(str(local_ds_dir))

    train_split = ds["train"] if "train" in ds else ds

    if str(args.dataset_config).lower() == "kn":
        # Sanity-check cached/loaded data to avoid accidental language-mismatch caches.
        sample_n = min(1000, len(train_split))
        sample_split = train_split.select(range(sample_n)) if sample_n > 0 else train_split
        kn_chars, total_chars = 0, 0
        for row in sample_split:
            _, tgt_text = extract_pair(row)
            for ch in tgt_text:
                total_chars += 1
                if "\u0c80" <= ch <= "\u0cff":
                    kn_chars += 1
        ratio = (kn_chars / max(1, total_chars))
        if ratio < 0.10:
            raise RuntimeError(
                "Loaded dataset appears non-Kannada (Kannada-char ratio {:.3f}). "
                "Clear wrong cache and retry with --save-dataset-to-disk. Checked path: {}".format(
                    ratio, local_ds_dir
                )
            )

    def map_extract(batch):
        size = len(next(iter(batch.values())))
        en_texts = []
        kn_texts = []
        for idx in range(size):
            row = {col: batch[col][idx] for col in batch}
            en, kn = extract_pair(row)
            en_texts.append(en)
            kn_texts.append(kn)
        return {"en": en_texts, "kn": kn_texts}

    keep_cols = ["en", "kn"]
    mapped = train_split.map(
        map_extract,
        batched=True,
        remove_columns=[c for c in train_split.column_names if c not in keep_cols],
        num_proc=args.num_proc,
        desc="Extracting EN-KN pairs",
    )

    filtered = mapped.filter(
        lambda en, kn: bool(en)
        and bool(kn)
        and len(en) >= args.min_char_len
        and len(kn) >= args.min_char_len
        and len(en) <= args.max_char_len
        and len(kn) <= args.max_char_len,
        input_columns=["en", "kn"],
        num_proc=args.num_proc,
        desc="Filtering empty/length-invalid rows",
    )

    if args.max_samples and len(filtered) > args.max_samples:
        filtered = filtered.shuffle(seed=args.seed).select(range(args.max_samples))

    split = filtered.train_test_split(test_size=args.val_fraction, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]

    print("Prepared pairs | train: {:,} | val: {:,}".format(len(train_ds), len(val_ds)))
    return train_ds, val_ds


def train_or_load_sentencepiece(train_ds, args):
    cache_dir = Path(args.cache_dir)
    tok_dir = cache_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = str(tok_dir / args.sp_model_name)
    model_file = model_prefix + ".model"

    if os.path.exists(model_file):
        print("Using existing tokenizer:", model_file)
        return model_file

    corpus_file = tok_dir / "sp_corpus.txt"
    print("Building SentencePiece corpus:", corpus_file)

    seen = 0
    with open(corpus_file, "w", encoding="utf-8") as handle:
        for row in train_ds:
            handle.write(row["en"] + "\n")
            handle.write(row["kn"] + "\n")
            seen += 1
            if seen >= args.tokenizer_pairs:
                break

    print("Training SentencePiece tokenizer on {:,} sentence pairs".format(seen))
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        input_sentence_size=min(4_000_000, seen * 2),
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        num_threads=os.cpu_count() or 4,
    )

    try:
        corpus_file.unlink()
    except FileNotFoundError:
        pass
    return model_file


def tokenize_and_pack(dataset, sp, max_seq_len, num_proc):
    def tok_batch(batch):
        src_encoded = sp.encode(batch["en"], out_type=int)
        tgt_encoded = sp.encode(batch["kn"], out_type=int)

        src_ids = []
        tgt_ids = []
        src_lens = []
        tgt_lens = []
        for src, tgt in zip(src_encoded, tgt_encoded):
            s = [BOS_ID] + src[: max_seq_len - 2] + [EOS_ID]
            t = [BOS_ID] + tgt[: max_seq_len - 2] + [EOS_ID]
            src_ids.append(s)
            tgt_ids.append(t)
            src_lens.append(len(s))
            tgt_lens.append(len(t))

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_len": src_lens,
            "tgt_len": tgt_lens,
        }

    return dataset.map(
        tok_batch,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )


class TokenizedNMTDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "src_ids": row["src_ids"],
            "tgt_ids": row["tgt_ids"],
            "src_len": int(row["src_len"]),
            "tgt_len": int(row["tgt_len"]),
        }


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, max_tokens=12000, shuffle=True):
        self.shuffle = shuffle
        self.max_tokens = max_tokens
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.order = np.argsort(self.lengths)
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        cur = []
        max_len = 0
        for idx in self.order:
            item_len = int(self.lengths[idx])
            next_max = max(max_len, item_len)
            if cur and (len(cur) + 1) * next_max > self.max_tokens:
                batches.append(cur)
                cur = [int(idx)]
                max_len = item_len
            else:
                cur.append(int(idx))
                max_len = next_max
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def collate_batch(items):
    src_max = max(len(x["src_ids"]) for x in items)
    tgt_max = max(len(x["tgt_ids"]) for x in items)

    src = torch.full((len(items), src_max), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(items), tgt_max), PAD_ID, dtype=torch.long)

    for row_idx, item in enumerate(items):
        src_ids = torch.tensor(item["src_ids"], dtype=torch.long)
        tgt_ids = torch.tensor(item["tgt_ids"], dtype=torch.long)
        src[row_idx, : src_ids.numel()] = src_ids
        tgt[row_idx, : tgt_ids.numel()] = tgt_ids

    return src, tgt


def noam_lr_lambda(step, warmup_steps):
    step = max(1, step)
    return min(step ** (-0.5), step * (warmup_steps ** -1.5)) * (warmup_steps ** 0.5)


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with amp_autocast(device, use_amp):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            n_tokens = int(tgt_out.ne(PAD_ID).sum().item())
            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def greedy_decode(model, sp, text, device, max_seq_len=128, max_gen_len=100):
    model.eval()

    src_ids = [BOS_ID] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [EOS_ID]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    memory, src_pad_mask = base_model.encode(src)

    tgt_ids = [BOS_ID]
    for _ in range(max_gen_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        next_logits = base_model.decode_step(tgt, memory, src_pad_mask)
        next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == EOS_ID:
            break
        tgt_ids.append(next_id)

    return sp.decode(tgt_ids[1:])


def evaluate_bleu(model, sp, val_text_ds, device, max_seq_len, max_samples):
    sample_size = min(len(val_text_ds), max_samples)
    if sample_size <= 0:
        return 0.0

    indices = np.random.choice(len(val_text_ds), size=sample_size, replace=False)
    hypotheses = []
    references = []

    for idx in indices:
        row = val_text_ds[int(idx)]
        pred = greedy_decode(model, sp, row["en"], device=device, max_seq_len=max_seq_len)
        hypotheses.append(pred)
        references.append(row["kn"])

    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


def main():
    parser = argparse.ArgumentParser(description="Train EN->KN translation model with Samanantar (HF datasets method).")
    parser.add_argument("--dataset-name", default="ai4bharat/samanantar")
    parser.add_argument("--dataset-config", default="kn")
    parser.add_argument("--cache-dir", default="translation/english_to_kannada/artifacts/english_to_kannada")
    parser.add_argument("--use-saved-dataset", action="store_true")
    parser.add_argument("--save-dataset-to-disk", action="store_true")
    parser.add_argument("--num-proc", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=600000)
    parser.add_argument("--val-fraction", type=float, default=0.001)
    parser.add_argument("--min-char-len", type=int, default=2)
    parser.add_argument("--max-char-len", type=int, default=500)

    parser.add_argument("--sp-model-name", default="english_to_kannada_unigram_v1")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--tokenizer-pairs", type=int, default=2500000)
    parser.add_argument("--max-seq-len", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=6)
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

    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device, gpu_only=args.gpu_only)
    use_amp = device.type == "cuda"

    cache_dir = Path(args.cache_dir)
    ckpt_dir = cache_dir / "checkpoints"
    if args.run_name:
        ckpt_dir = ckpt_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_text_ds, val_text_ds = load_and_prepare_dataset(args)
    sp_model_file = train_or_load_sentencepiece(train_text_ds, args)
    sp = spm.SentencePieceProcessor(model_file=sp_model_file)
    print("Tokenizer vocab size:", sp.get_piece_size())

    train_tok_ds = tokenize_and_pack(train_text_ds, sp, args.max_seq_len, args.num_proc)
    val_tok_ds = tokenize_and_pack(val_text_ds, sp, args.max_seq_len, args.num_proc)

    train_ds = TokenizedNMTDataset(train_tok_ds)
    val_ds = TokenizedNMTDataset(val_tok_ds)

    train_lengths = np.maximum(np.array(train_tok_ds["src_len"]), np.array(train_tok_ds["tgt_len"]))
    val_lengths = np.maximum(np.array(val_tok_ds["src_len"]), np.array(val_tok_ds["tgt_len"]))

    num_workers = max(1, min(4, os.cpu_count() or 1))
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_lengths, max_tokens=args.max_tokens_per_batch, shuffle=True),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(val_lengths, max_tokens=args.max_tokens_per_batch, shuffle=False),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    cfg = MODEL_CONFIG
    model = TransformerNMT(vocab_size=args.vocab_size, pad_id=PAD_ID, **cfg).to(device)

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

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as err:
            print("torch.compile unavailable:", err)

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
    print("Model: 10M | params: {:,}".format(param_count))
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
                    },
                    step_path,
                )
                print("Saved step checkpoint:", step_path)

            if args.max_train_steps > 0 and step >= args.max_train_steps:
                print("Reached max train steps ({}); stopping early.".format(args.max_train_steps))
                stop_training = True
                break

        train_loss = total_loss / max(1, total_tokens)
        val_loss = evaluate(model, val_loader, criterion, device=device, use_amp=use_amp)
        bleu = evaluate_bleu(
            model,
            sp,
            val_text_ds,
            device=device,
            max_seq_len=args.max_seq_len,
            max_samples=args.bleu_samples,
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


if __name__ == "__main__":
    main()