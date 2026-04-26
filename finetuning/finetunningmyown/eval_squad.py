import argparse
import collections
import re
import string
from pathlib import Path

import torch
from transformers import AutoTokenizer
from .adapters import CustomBertEncoderAdapter
from .reader import ReaderModel, load_squad_examples
from .test_reader import extract_answer
from tqdm import tqdm

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

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(ground_truth, prediction)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    package_root = Path(__file__).resolve().parents[1]
    package_candidate = package_root / path
    if package_candidate.exists():
        return package_candidate
    return candidate

def main():
    parser = argparse.ArgumentParser(description="Evaluate the finetuned reader on local SQuAD v1.1.")
    parser.add_argument("--dev-json", default="squad/dev-v1.1.json")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--encoder-checkpoint", default="pretrain/mlm_final/step_100000")
    parser.add_argument("--weights", default="finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model architecture
    encoder_ckpt = str(resolve_path(args.encoder_checkpoint))
    print(f"Loading tokenizer from {encoder_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_ckpt, use_fast=True)
    
    print(f"Loading model architecture...")
    encoder = CustomBertEncoderAdapter(checkpoint_dir=encoder_ckpt)
    model = ReaderModel(encoder=encoder).to(device)

    # Load finetuned weights
    best_ckpt = str(resolve_path(args.weights))
    print(f"Loading finetuned weights from {best_ckpt}...")
    payload = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dev_json = str(resolve_path(args.dev_json))
    print(f"Loading SQuAD v1.1 validation examples from {dev_json}...")
    examples = load_squad_examples(dev_json)
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    print(f"Evaluating on {len(examples)} examples...")
    
    f1_scores = []
    em_scores = []
    
    # Let's print a few predictions to show tests
    num_to_print = 3
    printed = 0

    for idx, example in enumerate(tqdm(examples)):
        question = example.question
        context = example.context
        answers = [example.answer_text]
            
        pred_answer, score = extract_answer(question, context, model, tokenizer, device=device)
        
        em = metric_max_over_ground_truths(compute_exact, pred_answer, answers)
        f1 = metric_max_over_ground_truths(compute_f1, pred_answer, answers)
        
        em_scores.append(em)
        f1_scores.append(f1)
        
        if printed < num_to_print:
            print(f"\nExample {idx+1}")
            print(f"Q: {question}")
            print(f"A (Pred): '{pred_answer}'")
            print(f"A (Gold): {answers}")
            print(f"F1: {f1:.3f}, EM: {em}")
            printed += 1

    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0.0
    
    print(f"\n--- Results on {len(examples)} examples ---")
    print(f"Exact Match (EM): {avg_em:.2f}")
    print(f"F1 Score: {avg_f1:.2f}")

if __name__ == "__main__":
    main()
