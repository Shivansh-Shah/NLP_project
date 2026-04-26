import argparse
import re
from pathlib import Path

import sentencepiece as spm
import torch

try:
    from .model_scratch import TransformerScratch
    from .runtime import maybe_quantize_for_cpu, resolve_device
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_scratch import TransformerScratch
    from runtime import maybe_quantize_for_cpu, resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_sp_model_path(sp_model_file):
    sp_path = Path(sp_model_file)
    if sp_path.is_absolute() and sp_path.exists():
        return sp_path
    if sp_path.exists():
        return sp_path.resolve()

    repo_candidate = REPO_ROOT / sp_path
    if repo_candidate.exists():
        return repo_candidate

    raise FileNotFoundError("SentencePiece model not found: {}".format(sp_model_file))


def load_runtime(checkpoint_path, device, cpu_quantize=False):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    sp = spm.SentencePieceProcessor(model_file=str(resolve_sp_model_path(checkpoint["sp_model_file"])))

    model = TransformerScratch(
        vocab_size=checkpoint["vocab_size"],
        pad_id=checkpoint.get("special_ids", {}).get("pad", 0),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = maybe_quantize_for_cpu(model, enable_quantization=cpu_quantize, device=device)
    model.eval()
    return model, sp, checkpoint


@torch.no_grad()
def greedy_translate(model, sp, text, special_ids, max_seq_len, max_gen_len, min_gen_len, repetition_penalty, device):
    bos_id = special_ids.get("bos", 2)
    eos_id = special_ids.get("eos", 3)
    pad_id = special_ids.get("pad", 0)
    unk_id = special_ids.get("unk", 1)

    src_ids = [bos_id] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    memory, src_mask = model.encode(src)
    tgt_ids = [bos_id]

    for _ in range(max_gen_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        next_logits = model.decode_step(tgt, memory, src_mask).squeeze(0)

        # Suppress UNK output and reduce degenerative loops.
        next_logits[unk_id] = -1e9
        if repetition_penalty > 1.0 and len(tgt_ids) > 1:
            for token_id in set(tgt_ids[1:]):
                next_logits[token_id] = next_logits[token_id] / repetition_penalty
        if len(tgt_ids) - 1 < min_gen_len:
            next_logits[eos_id] = -1e9

        next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == eos_id or next_id == pad_id:
            break
        tgt_ids.append(next_id)

    return sp.decode(tgt_ids[1:])


def split_sentences(text):
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else [""]

    sentences = []
    for paragraph in paragraphs:
        pieces = re.split(r"(?<=[.!?])\s+", paragraph)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def split_overlong_sentence(sentence, sp, max_src_pieces):
    words = sentence.split()
    if not words:
        return [sentence]

    chunks = []
    current_words = []

    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)
        if current_words and len(sp.encode(candidate, out_type=int)) > max_src_pieces:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def segment_text(text, sp, max_seq_len):
    max_src_pieces = max_seq_len - 2
    segments = []
    for sentence in split_sentences(text):
        if len(sp.encode(sentence, out_type=int)) <= max_src_pieces:
            segments.append(sentence)
        else:
            segments.extend(split_overlong_sentence(sentence, sp, max_src_pieces))
    return [seg for seg in segments if seg.strip()]


def main():
    parser = argparse.ArgumentParser(description="Translate with a scratch Transformer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--cpu-quantize", action="store_true", help="Apply dynamic int8 quantization when running on CPU.")
    parser.add_argument("--max-gen-len", type=int, default=100)
    parser.add_argument("--min-gen-len", type=int, default=6)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--segment-long-text", action="store_true", default=True)
    parser.add_argument("--no-segment-long-text", dest="segment_long_text", action="store_false")
    args = parser.parse_args()

    device = resolve_device(args.device, gpu_only=args.gpu_only)

    model, sp, checkpoint = load_runtime(args.checkpoint, device, cpu_quantize=args.cpu_quantize)
    special_ids = checkpoint.get("special_ids", {"pad": 0, "unk": 1, "bos": 2, "eos": 3})
    max_seq_len = int(checkpoint.get("max_seq_len", 128))
    source_lang = str(checkpoint.get("source_lang", "en")).upper()
    target_lang = str(checkpoint.get("target_lang", "kn")).upper()

    if args.segment_long_text:
        text_segments = segment_text(args.text, sp, max_seq_len=max_seq_len)
    else:
        text_segments = [args.text]

    pred_parts = []
    for text_segment in text_segments:
        pred_parts.append(
            greedy_translate(
                model,
                sp,
                text_segment,
                special_ids=special_ids,
                max_seq_len=max_seq_len,
                max_gen_len=args.max_gen_len,
                min_gen_len=args.min_gen_len,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
        )
    pred = " ".join(part.strip() for part in pred_parts if part.strip())

    print("{}:".format(source_lang), args.text)
    print("{}:".format(target_lang), pred)


if __name__ == "__main__":
    main()
