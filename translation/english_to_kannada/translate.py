import argparse
import warnings
from pathlib import Path

import sentencepiece as spm
import torch

try:
    from .model import TransformerNMT
except ImportError:  # pragma: no cover - direct script execution fallback
    from model import TransformerNMT

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
except Exception:
    sanscript = None
    indic_transliterate = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_sp_model_path(sp_model_file: str) -> Path:
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

    raise FileNotFoundError(f"SentencePiece model not found: {sp_model_file}")


def load_runtime(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    sp = spm.SentencePieceProcessor(model_file=str(_resolve_sp_model_path(checkpoint["sp_model_file"])))

    model = TransformerNMT(
        vocab_size=checkpoint["vocab_size"],
        pad_id=checkpoint.get("special_ids", {}).get("pad", 0),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp, checkpoint


@torch.no_grad()
def greedy_translate(
    model,
    sp,
    text,
    special_ids,
    max_seq_len,
    max_gen_len,
    min_gen_len,
    repetition_penalty,
    device,
):
    bos_id = special_ids.get("bos", 2)
    eos_id = special_ids.get("eos", 3)
    pad_id = special_ids.get("pad", 0)
    unk_id = special_ids.get("unk", 1)

    src_ids = [bos_id] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    memory, src_pad_mask = model.encode(src)
    tgt_ids = [bos_id]
    punctuation_ids = {
        i for i in range(sp.get_piece_size())
        if sp.id_to_piece(i) in {".", ",", "!", "?", "...", "..", "▁."}
    }
    seen_bigrams = set()

    for _ in range(max_gen_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        next_logits = model.decode_step(tgt, memory, src_pad_mask).squeeze(0)

        # Avoid emitting UNK token, which decodes as '⁇' and hides script quality.
        next_logits[unk_id] = -1e9

        # Penalize recently used tokens to reduce repetitive loops.
        if repetition_penalty > 1.0 and len(tgt_ids) > 1:
            unique_tokens = set(tgt_ids[1:])
            for tok_id in unique_tokens:
                next_logits[tok_id] = next_logits[tok_id] / repetition_penalty

        # Block immediate 3+ token repetition (e.g. "ನಾನು ನಾನು ನಾನು ...").
        if len(tgt_ids) >= 3 and tgt_ids[-1] == tgt_ids[-2]:
            next_logits[tgt_ids[-1]] = -1e9

        # Simple no-repeat bigram constraint.
        if len(tgt_ids) >= 2:
            prev = tgt_ids[-1]
            for cand in range(sp.get_piece_size()):
                if (prev, cand) in seen_bigrams:
                    next_logits[cand] = -1e9

        # Prevent immediate EOS and punctuation-only collapse on undertrained checkpoints.
        if len(tgt_ids) - 1 < min_gen_len:
            next_logits[eos_id] = -1e9
            for pid in punctuation_ids:
                next_logits[pid] = -1e9

        next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == eos_id or next_id == pad_id:
            break
        if len(tgt_ids) >= 1:
            seen_bigrams.add((tgt_ids[-1], next_id))
        tgt_ids.append(next_id)
    return sp.decode(tgt_ids[1:])


def main():
    parser = argparse.ArgumentParser(description="Translate with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-gen-len", type=int, default=100)
    parser.add_argument("--min-gen-len", type=int, default=6)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--show-unicode", action="store_true", help="Also print output as unicode escapes.")
    parser.add_argument("--romanize", action="store_true", help="Romanize output when target language is Kannada.")
    parser.add_argument(
        "--romanize-scheme",
        choices=["itrans", "hk"],
        default="hk",
        help="Romanization scheme for KN output (default: hk for cleaner ASCII).",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, sp, checkpoint = load_runtime(args.checkpoint, device)
    special_ids = checkpoint.get("special_ids", {"pad": 0, "unk": 1, "bos": 2, "eos": 3})
    max_seq_len = int(checkpoint.get("max_seq_len", 128))
    source_lang = str(checkpoint.get("source_lang", "en")).upper()
    target_lang = str(checkpoint.get("target_lang", "kn")).upper()

    pred = greedy_translate(
        model,
        sp,
        args.text,
        special_ids=special_ids,
        max_seq_len=max_seq_len,
        max_gen_len=args.max_gen_len,
        min_gen_len=args.min_gen_len,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print("{}:".format(source_lang), args.text)
    print("{}:".format(target_lang), pred)
    if args.show_unicode:
        escaped = pred.encode("unicode_escape").decode("ascii")
        print("{}_UNICODE:".format(target_lang), escaped)
    if args.romanize:
        if target_lang == "KN" and indic_transliterate is not None and sanscript is not None:
            target_scheme = sanscript.HK if args.romanize_scheme == "hk" else sanscript.ITRANS
            roman = indic_transliterate(pred, sanscript.KANNADA, target_scheme)
            print("KN_ROMAN:", roman)
        elif target_lang != "KN":
            print("KN_ROMAN:", "(romanization is only applicable when target language is Kannada)")
        else:
            print("KN_ROMAN:", "(install indic-transliteration to enable romanization)")


if __name__ == "__main__":
    main()