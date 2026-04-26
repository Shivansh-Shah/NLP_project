import argparse
import collections
import re
import string

import torch
from transformers import AutoTokenizer
from .adapters import CustomBertEncoderAdapter
from .reader import ReaderModel


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def highlight_answer(context: str, answer: str) -> str:
    if not answer:
        return context
    lower_context = context.lower()
    lower_answer = answer.lower()
    start = lower_context.find(lower_answer)
    if start == -1:
        return context
    end = start + len(answer)
    return context[:start] + "[[" + context[start:end] + "]]" + context[end:]

@torch.no_grad()
def extract_answer(
    question: str,
    context: str,
    model,
    tokenizer,
    max_len: int = 384,
    max_span: int = 30,
    device="cpu",
):
    device = torch.device(device)
    model.eval()

    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        
    offsets = enc["offset_mapping"][0].tolist()

    seq_ids = enc.sequence_ids(0) if hasattr(enc, "sequence_ids") else None
    if seq_ids is None:
        return "", float("-inf")

    start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    start_logits = start_logits[0]
    end_logits = end_logits[0]

    context_mask = torch.tensor([1 if sid == 1 else 0 for sid in seq_ids], device=device, dtype=torch.bool)
    if context_mask.sum().item() == 0:
        return "", float("-inf")

    neg_inf = torch.tensor(-1e9, device=device)
    start_logits = torch.where(context_mask, start_logits, neg_inf)
    end_logits = torch.where(context_mask, end_logits, neg_inf)

    best_score = float("-inf")
    best_s = None
    best_e = None
    seq_len = int(input_ids.size(1))
    for i in range(seq_len):
        if not context_mask[i]:
            continue
        jmax = min(seq_len, i + max_span)
        for j in range(i, jmax):
            if not context_mask[j]:
                continue
            score = float(start_logits[i].item() + end_logits[j].item())
            if score > best_score:
                best_score = score
                best_s, best_e = i, j

    if best_s is None or best_e is None:
        return "", float("-inf")

    start_char = offsets[best_s][0]
    end_char = offsets[best_e][1]
    if start_char is None or end_char is None or end_char <= start_char:
        return "", float("-inf")
    return context[start_char:end_char].strip(), float(best_score)


def load_model(device: str, encoder_ckpt: str, weights_path: str):
    tokenizer = AutoTokenizer.from_pretrained(encoder_ckpt, use_fast=True)
    encoder = CustomBertEncoderAdapter(checkpoint_dir=encoder_ckpt)
    model = ReaderModel(encoder=encoder).to(device)
    payload = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser(description="Run a finetuned reader on one context or a built-in demo.")
    parser.add_argument("--question", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--gold-answer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder-checkpoint", default="pretrain/mlm_final/step_100000")
    parser.add_argument("--weights", default="finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    tokenizer, model = load_model(device, args.encoder_checkpoint, args.weights)
    print("Model loaded successfully!\n")

    if args.question and args.context:
        answer, score = extract_answer(args.question, args.context, model, tokenizer, device=device)
        print(f"Q: {args.question}")
        print(f"A: {answer} (score: {score:.2f})")
        print(f"Context: {highlight_answer(args.context, answer)}")
        if args.gold_answer is not None:
            em = compute_exact(args.gold_answer, answer)
            f1 = compute_f1(args.gold_answer, answer)
            print(f"Gold: {args.gold_answer}")
            print(f"EM: {em}")
            print(f"F1: {f1:.3f}")
        return

    context = "The Apollo 11 mission was the first spaceflight that landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969."

    questions = [
        "Who was the commander of Apollo 11?",
        "When did Apollo 11 land on the Moon?",
        "What was the name of the lunar module?",
    ]

    print(f"Context: {context}\n")
    for q in questions:
        answer, score = extract_answer(q, context, model, tokenizer, device=device)
        print(f"Q: {q}")
        print(f"A: {answer} (score: {score:.2f})")
        print(f"Retrieved context: {highlight_answer(context, answer)}\n")

if __name__ == "__main__":
    main()
