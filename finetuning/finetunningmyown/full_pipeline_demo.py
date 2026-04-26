"""Full Kannada QA pipeline: KN question -> EN QA -> KN answer.

This script wires three stages:
1) Kannada question to English translation (KN->EN model)
2) English extractive QA on the provided context
3) English answer back to Kannada translation (EN->KN model)
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import sentencepiece as spm
import torch

from .test_reader import extract_answer, highlight_answer, load_model
from translation.english_to_kannada.model import TransformerNMT


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _torch_load(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _resolve_sp_model_path(checkpoint_path: Path, sp_model_file: str) -> Path:
    sp_path = Path(sp_model_file)
    if sp_path.is_absolute() and sp_path.exists():
        return sp_path
    if sp_path.exists():
        return sp_path.resolve()

    repo_candidate = REPO_ROOT / sp_path
    if repo_candidate.exists():
        return repo_candidate

    if sp_path.parts and sp_path.parts[0] == "artifacts":
        legacy_text = sp_path.as_posix().split("artifacts/", 1)[-1]
        legacy_targets = {
            "translation_en_kn_v2/tokenizer/kn_en_unigram_v2.model": [
                REPO_ROOT / "translation" / "kannada_to_english" / "kannada_to_english_v2" / "tokenizer" / "kannada_to_english_unigram_v2.model",
            ],
            "translation_en_kn/tokenizer/en_kn_unigram_v1.model": [
                REPO_ROOT / "translation" / "english_to_kannada" / "artifacts" / "english_to_kannada" / "tokenizer" / "english_to_kannada_unigram_v1.model",
            ],
        }
        for suffix, candidates in legacy_targets.items():
            if legacy_text.endswith(suffix):
                for alt_candidate in candidates:
                    if alt_candidate.exists():
                        return alt_candidate

    for parent in checkpoint_path.resolve().parents:
        candidate = parent / sp_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"SentencePiece model not found: {sp_model_file} (checkpoint: {checkpoint_path})"
    )


def load_translation_runtime(checkpoint_path: str, device: torch.device):
    ckpt_path = resolve_path(checkpoint_path)
    checkpoint = _torch_load(ckpt_path, device)
    sp_path = _resolve_sp_model_path(ckpt_path, checkpoint["sp_model_file"])
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))

    model = TransformerNMT(
        vocab_size=int(checkpoint["vocab_size"]),
        pad_id=int(checkpoint.get("special_ids", {}).get("pad", 0)),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp, checkpoint


@torch.no_grad()
def translate_text(
    model,
    sp,
    checkpoint: Dict,
    text: str,
    device: torch.device,
    max_gen_len: int = 100,
    repetition_penalty: float = 1.2,
    min_gen_len: int = 6,
) -> str:
    special_ids = checkpoint.get("special_ids", {"pad": 0, "unk": 1, "bos": 2, "eos": 3})
    bos_id = int(special_ids.get("bos", 2))
    eos_id = int(special_ids.get("eos", 3))
    pad_id = int(special_ids.get("pad", 0))
    unk_id = int(special_ids.get("unk", 1))
    max_seq_len = int(checkpoint.get("max_seq_len", 128))

    src_ids = [bos_id] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    memory, src_pad_mask = model.encode(src)
    tgt_ids = [bos_id]
    seen_bigrams = set()

    punctuation_ids = {
        i for i in range(sp.get_piece_size()) if sp.id_to_piece(i) in {".", ",", "!", "?", "...", "..", "▁."}
    }

    for _ in range(max_gen_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        next_logits = model.decode_step(tgt, memory, src_pad_mask).squeeze(0)
        next_logits[unk_id] = -1e9

        if repetition_penalty > 1.0 and len(tgt_ids) > 1:
            unique_tokens = set(tgt_ids[1:])
            for tok_id in unique_tokens:
                next_logits[tok_id] = next_logits[tok_id] / repetition_penalty

        if len(tgt_ids) >= 3 and tgt_ids[-1] == tgt_ids[-2]:
            next_logits[tgt_ids[-1]] = -1e9

        if len(tgt_ids) - 1 < min_gen_len:
            next_logits[eos_id] = -1e9
            for pid in punctuation_ids:
                next_logits[pid] = -1e9

        if len(tgt_ids) >= 2:
            prev = tgt_ids[-1]
            for cand in range(sp.get_piece_size()):
                if (prev, cand) in seen_bigrams:
                    next_logits[cand] = -1e9

        next_id = int(next_logits.argmax(dim=-1).item())
        if next_id in (eos_id, pad_id):
            break

        if len(tgt_ids) >= 1:
            seen_bigrams.add((tgt_ids[-1], next_id))
        tgt_ids.append(next_id)

    return sp.decode(tgt_ids[1:]).strip()


def extract_context_excerpt(context_en: str, answer_en: str, window_chars: int = 220) -> str:
    if not context_en:
        return ""
    if not answer_en:
        return context_en[:window_chars].strip()

    lower_context = context_en.lower()
    lower_answer = answer_en.lower().strip()
    start = lower_context.find(lower_answer)

    if start == -1:
        return context_en[:window_chars].strip()

    end = start + len(answer_en)
    left = max(0, start - window_chars // 2)
    right = min(len(context_en), end + window_chars // 2)

    snippet = context_en[left:right].strip()
    if left > 0:
        snippet = "... " + snippet
    if right < len(context_en):
        snippet = snippet + " ..."
    return snippet


def run_pipeline(
    context_en: str,
    question_kn: str,
    kn_en_ckpt: str,
    en_kn_ckpt: str,
    reader_encoder_ckpt: str,
    reader_weights: str,
    device: str,
) -> Dict[str, str]:
    torch_device = torch.device(device)

    kn_en_model, kn_en_sp, kn_en_meta = load_translation_runtime(kn_en_ckpt, torch_device)
    en_kn_model, en_kn_sp, en_kn_meta = load_translation_runtime(en_kn_ckpt, torch_device)
    tokenizer, reader_model = load_model(torch_device, str(resolve_path(reader_encoder_ckpt)), str(resolve_path(reader_weights)))

    question_en = translate_text(kn_en_model, kn_en_sp, kn_en_meta, question_kn, torch_device)
    answer_en, score = extract_answer(question_en, context_en, reader_model, tokenizer, device=device)
    answer_kn = translate_text(en_kn_model, en_kn_sp, en_kn_meta, answer_en, torch_device) if answer_en else ""
    retrieved_context_en = extract_context_excerpt(context_en, answer_en)

    return {
        "question_kn": question_kn,
        "question_en": question_en,
        "context_en": context_en,
        "answer_en": answer_en,
        "answer_kn": answer_kn,
        "answer_score": f"{score:.2f}",
        "highlighted_context_en": highlight_answer(retrieved_context_en, answer_en),
    }


def main():
    parser = argparse.ArgumentParser(description="Full KN->EN QA->KN pipeline demo")
    parser.add_argument("--context", required=True, help="English context text")
    parser.add_argument("--question-kn", required=True, help="Kannada question")
    parser.add_argument(
        "--kn-en-checkpoint",
        default="translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt",
        help="Checkpoint for Kannada->English translation",
    )
    parser.add_argument(
        "--en-kn-checkpoint",
        default="translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt",
        help="Checkpoint for English->Kannada translation",
    )
    parser.add_argument(
        "--reader-encoder-checkpoint",
        default="pretrain/mlm_final/step_100000",
        help="Tokenizer/encoder checkpoint for the QA reader",
    )
    parser.add_argument(
        "--reader-weights",
        default="finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt",
        help="Finetuned QA reader weights",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result = run_pipeline(
        context_en=args.context,
        question_kn=args.question_kn,
        kn_en_ckpt=args.kn_en_checkpoint,
        en_kn_ckpt=args.en_kn_checkpoint,
        reader_encoder_ckpt=args.reader_encoder_checkpoint,
        reader_weights=args.reader_weights,
        device=device,
    )

    print("Question (KN):", result["question_kn"])
    print("Question (EN):", result["question_en"])
    print("Retrieved Context (EN):", result["highlighted_context_en"])
    print("Answer (EN):", result["answer_en"])
    print("Answer (KN):", result["answer_kn"])
    print("Answer Score:", result["answer_score"])


if __name__ == "__main__":
    main()