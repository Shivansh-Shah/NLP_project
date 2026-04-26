import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

@dataclass
class SquadExample:
    qid: str
    question: str
    context: str
    answer_text: str
    answer_start: int

def load_squad_examples(path) -> List[SquadExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[SquadExample] = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                if not answers:
                    continue
                ans = answers[0]
                out.append(
                    SquadExample(
                        qid=str(qa.get("id", "")),
                        question=qa["question"],
                        context=context,
                        answer_text=ans["text"],
                        answer_start=int(ans["answer_start"]),
                    )
                )
    return out

class ReaderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.start_head = nn.Linear(encoder.hidden_size, 1)
        self.end_head = nn.Linear(encoder.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        hidden = self.encoder.encode_tokens(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)
        return start_logits, end_logits

def build_feature(tokenizer, ex: SquadExample, max_length: int) -> Optional[Dict[str, torch.Tensor]]:
    enc = tokenizer(
        ex.question,
        ex.context,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True,
        padding="max_length",
    )

    offsets = enc["offset_mapping"]
    seq_ids = enc.sequence_ids() if hasattr(enc, "sequence_ids") else None
    if seq_ids is None:
        return None

    answer_start = ex.answer_start
    answer_end = answer_start + len(ex.answer_text)

    token_start = None
    token_end = None
    for i, (sid, off) in enumerate(zip(seq_ids, offsets)):
        if sid != 1:
            continue
        if token_start is None and off[0] <= answer_start < off[1]:
            token_start = i
        if off[0] < answer_end <= off[1]:
            token_end = i
            break

    if token_start is None or token_end is None:
        return None

    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "token_type_ids": torch.tensor(enc.get("token_type_ids", [0]*len(enc["input_ids"])), dtype=torch.long),
        "start_positions": torch.tensor(token_start, dtype=torch.long),
        "end_positions": torch.tensor(token_end, dtype=torch.long),
    }
