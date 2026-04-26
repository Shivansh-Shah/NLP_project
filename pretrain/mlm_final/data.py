import random

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


class StreamingTextDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=128, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        self.stream = self.stream.shuffle(seed=seed, buffer_size=20000)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        token_buffer = []
        _ = random.Random(self.seed + worker_id)
        if num_workers > 1 and hasattr(self.stream, "shard"):
            stream = self.stream.shard(num_shards=num_workers, index=worker_id)
        else:
            stream = self.stream
        for ex in stream:
            if num_workers > 1:
                ex_idx = getattr(self, "_worker_example_idx", 0)
                self._worker_example_idx = ex_idx + 1
                if (ex_idx % num_workers) != worker_id:
                    continue
            text = ex.get("text", "")
            if not text or len(text) < 20:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) == 0:
                continue
            token_buffer.extend(ids)
            while len(token_buffer) >= (self.max_len - 2):
                chunk = token_buffer[: self.max_len - 2]
                token_buffer = token_buffer[self.max_len - 2 :]
                input_ids = [self.cls_id] + chunk + [self.sep_id]
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }


def collate_mlm(batch, tokenizer, mlm_prob=0.15):
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        length = len(x["input_ids"])
        input_ids[i, :length] = torch.tensor(x["input_ids"], dtype=torch.long)
        attention_mask[i, :length] = torch.tensor(x["attention_mask"], dtype=torch.long)
        token_type_ids[i, :length] = torch.tensor(x["token_type_ids"], dtype=torch.long)

    labels = torch.full_like(input_ids, -100)
    special_mask = (
        (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.pad_token_id)
    )
    prob = torch.full(input_ids.shape, mlm_prob)
    prob.masked_fill_(special_mask, 0.0)
    masked = torch.bernoulli(prob).bool()
    labels[masked] = input_ids[masked]

    replace_prob = torch.rand(input_ids.shape)
    mask80 = masked & (replace_prob < 0.8)
    input_ids[mask80] = mask_id
    rand10 = masked & (replace_prob >= 0.8) & (replace_prob < 0.9)
    random_words = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=torch.long)
    input_ids[rand10] = random_words[rand10]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }