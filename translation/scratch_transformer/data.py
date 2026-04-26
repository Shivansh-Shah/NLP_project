import os
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, Sampler


def kn_char_ratio(text):
    if not text:
        return 0.0
    total = len(text)
    kn_chars = sum(1 for ch in text if "\u0c80" <= ch <= "\u0cff")
    return kn_chars / max(1, total)


def to_en_kn_pair(example):
    translation = example.get("translation")
    if isinstance(translation, dict):
        en = translation.get("en")
        kn = translation.get("kn")
        if en and kn:
            return str(en).strip(), str(kn).strip()

    en = example.get("en")
    kn = example.get("kn")
    if en and kn:
        return str(en).strip(), str(kn).strip()

    src = example.get("src", example.get("source"))
    tgt = example.get("tgt", example.get("target"))
    if src is None or tgt is None:
        return "", ""

    src = str(src).strip()
    tgt = str(tgt).strip()
    if not src or not tgt:
        return "", ""

    if kn_char_ratio(src) >= kn_char_ratio(tgt):
        return tgt, src
    return src, tgt


def extract_pair(example, source_lang, target_lang):
    en_text, kn_text = to_en_kn_pair(example)
    if not en_text or not kn_text:
        return "", ""

    if source_lang == "en" and target_lang == "kn":
        return en_text, kn_text
    if source_lang == "kn" and target_lang == "en":
        return kn_text, en_text

    raise ValueError("Only en<->kn directions are supported for this trainer.")


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

    def map_extract(batch):
        size = len(next(iter(batch.values())))
        src_texts = []
        tgt_texts = []
        for idx in range(size):
            row = {col: batch[col][idx] for col in batch}
            src, tgt = extract_pair(row, source_lang=args.source_lang, target_lang=args.target_lang)
            src_texts.append(src)
            tgt_texts.append(tgt)
        return {"src": src_texts, "tgt": tgt_texts}

    mapped = train_split.map(
        map_extract,
        batched=True,
        remove_columns=train_split.column_names,
        num_proc=args.num_proc,
        desc="Extracting {}-{} pairs".format(args.source_lang.upper(), args.target_lang.upper()),
    )

    filtered = mapped.filter(
        lambda src, tgt: bool(src)
        and bool(tgt)
        and len(src) >= args.min_char_len
        and len(tgt) >= args.min_char_len
        and len(src) <= args.max_char_len
        and len(tgt) <= args.max_char_len,
        input_columns=["src", "tgt"],
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


def train_or_load_sentencepiece(train_ds, args, pad_id, unk_id, bos_id, eos_id):
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
            handle.write(row["src"] + "\n")
            handle.write(row["tgt"] + "\n")
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
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
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


def tokenize_and_pack(dataset, sp, max_seq_len, num_proc, bos_id, eos_id):
    def tok_batch(batch):
        src_encoded = sp.encode(batch["src"], out_type=int)
        tgt_encoded = sp.encode(batch["tgt"], out_type=int)

        src_ids = []
        tgt_ids = []
        src_lens = []
        tgt_lens = []
        for src, tgt in zip(src_encoded, tgt_encoded):
            s = [bos_id] + src[: max_seq_len - 2] + [eos_id]
            t = [bos_id] + tgt[: max_seq_len - 2] + [eos_id]
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


def collate_batch(items, pad_id):
    src_max = max(len(x["src_ids"]) for x in items)
    tgt_max = max(len(x["tgt_ids"]) for x in items)

    src = torch.full((len(items), src_max), pad_id, dtype=torch.long)
    tgt = torch.full((len(items), tgt_max), pad_id, dtype=torch.long)

    for row_idx, item in enumerate(items):
        src_ids = torch.tensor(item["src_ids"], dtype=torch.long)
        tgt_ids = torch.tensor(item["tgt_ids"], dtype=torch.long)
        src[row_idx, : src_ids.numel()] = src_ids
        tgt[row_idx, : tgt_ids.numel()] = tgt_ids

    return src, tgt
