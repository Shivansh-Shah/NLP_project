from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch

from finetuning.finetunningmyown.full_pipeline_demo import (
    load_translation_runtime as load_legacy_translation_runtime,
    resolve_path as resolve_legacy_path,
    translate_text as legacy_translate_text,
)
from finetuning.finetunningmyown.test_reader import extract_answer, highlight_answer, load_model
from translation.scratch_transformer.translate_scratch import (
    greedy_translate as scratch_greedy_translate,
    load_runtime as load_scratch_translation_runtime,
    segment_text as scratch_segment_text,
)


ROOT = Path(__file__).resolve().parents[1]

BACKEND_CHOICES = ("scratch", "legacy", "nn_transformer")
DEVICE_CHOICES = ("auto", "cuda", "cpu")

DEFAULT_PATHS = {
    "scratch_kn_en": "translation/scratch_transformer/artifacts/kannada_to_english_scratch/checkpoints/kn_en_40m_v1_20260422/best.pt",
    "scratch_en_kn": "translation/scratch_transformer/artifacts/english_to_kannada_scratch/checkpoints/en_kn_40m_v1_20260422/best.pt",
    "legacy_kn_en": "translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt",
    "legacy_en_kn": "translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt",
    "reader_encoder": "pretrain/mlm_final/step_100000",
    "reader_weights": "finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt",
}


@dataclass(frozen=True)
class PipelineSettings:
    device: str = "auto"
    cpu_quantize: bool = False
    kn_en_backend: str = "scratch"
    en_kn_backend: str = "scratch"
    kn_en_checkpoint: str = DEFAULT_PATHS["scratch_kn_en"]
    en_kn_checkpoint: str = DEFAULT_PATHS["scratch_en_kn"]
    reader_encoder_checkpoint: str = DEFAULT_PATHS["reader_encoder"]
    reader_weights: str = DEFAULT_PATHS["reader_weights"]
    max_gen_len: int = 160
    min_gen_len: int = 6
    repetition_penalty: float = 1.2


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def _validate_backend(backend: str) -> None:
    if backend not in BACKEND_CHOICES:
        raise ValueError("Unsupported backend '{}'. Use one of {}.".format(backend, BACKEND_CHOICES))


def _normalize_backend(backend: str) -> str:
    _validate_backend(backend)
    if backend in ("legacy", "nn_transformer"):
        return "legacy"
    return "scratch"


def _display_backend(backend: str) -> str:
    if backend == "legacy":
        return "nn_transformer"
    return backend


def validate_settings_paths(settings: PipelineSettings) -> Dict[str, str]:
    results = {}
    for label, path_text in {
        "KN->EN checkpoint": settings.kn_en_checkpoint,
        "EN->KN checkpoint": settings.en_kn_checkpoint,
        "Reader encoder": settings.reader_encoder_checkpoint,
        "Reader weights": settings.reader_weights,
    }.items():
        path = _resolve_repo_path(path_text)
        results[label] = "ok" if path.exists() else "missing"
    return results


def _load_translation_runtime(backend: str, checkpoint_path: str, device: torch.device, cpu_quantize: bool) -> Dict[str, object]:
    normalized_backend = _normalize_backend(backend)
    ckpt_path = _resolve_repo_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))

    if normalized_backend == "legacy":
        model, sp, meta = load_legacy_translation_runtime(str(ckpt_path), device)
    else:
        model, sp, meta = load_scratch_translation_runtime(str(ckpt_path), device, cpu_quantize=cpu_quantize)

    return {
        "backend": _display_backend(normalized_backend),
        "model": model,
        "sp": sp,
        "meta": meta,
    }


def _translate(runtime: Dict[str, object], text: str, device: torch.device, settings: PipelineSettings) -> str:
    backend = str(runtime["backend"])
    model = runtime["model"]
    sp = runtime["sp"]
    meta = runtime["meta"]

    if backend == "nn_transformer":
        return legacy_translate_text(model, sp, meta, text, device).strip()

    special_ids = meta.get("special_ids", {"pad": 0, "unk": 1, "bos": 2, "eos": 3})
    max_seq_len = int(meta.get("max_seq_len", 128))
    segments = scratch_segment_text(text, sp, max_seq_len=max_seq_len)
    outputs = []
    for segment in segments:
        translated = scratch_greedy_translate(
            model=model,
            sp=sp,
            text=segment,
            special_ids=special_ids,
            max_seq_len=max_seq_len,
            max_gen_len=settings.max_gen_len,
            min_gen_len=settings.min_gen_len,
            repetition_penalty=settings.repetition_penalty,
            device=device,
        ).strip()
        if translated:
            outputs.append(translated)
    return " ".join(outputs).strip()


@lru_cache(maxsize=8)
def _load_pipeline_resources_cached(
    device_choice: str,
    cpu_quantize: bool,
    kn_en_backend: str,
    en_kn_backend: str,
    kn_en_checkpoint: str,
    en_kn_checkpoint: str,
    reader_encoder_checkpoint: str,
    reader_weights: str,
) -> Dict[str, object]:
    device = _resolve_device(device_choice)

    kn_en_runtime = _load_translation_runtime(kn_en_backend, kn_en_checkpoint, device, cpu_quantize)
    en_kn_runtime = _load_translation_runtime(en_kn_backend, en_kn_checkpoint, device, cpu_quantize)

    tokenizer, reader_model = load_model(
        str(device),
        str(resolve_legacy_path(reader_encoder_checkpoint)),
        str(resolve_legacy_path(reader_weights)),
    )

    return {
        "device": device,
        "kn_en_runtime": kn_en_runtime,
        "en_kn_runtime": en_kn_runtime,
        "tokenizer": tokenizer,
        "reader_model": reader_model,
    }


def clear_pipeline_cache() -> None:
    _load_pipeline_resources_cached.cache_clear()


def run_pipeline_on_context(context_en: str, question_kn: str, settings: PipelineSettings | None = None) -> Dict[str, str]:
    active_settings = settings or PipelineSettings()
    resources = _load_pipeline_resources_cached(
        active_settings.device,
        active_settings.cpu_quantize,
        active_settings.kn_en_backend,
        active_settings.en_kn_backend,
        active_settings.kn_en_checkpoint,
        active_settings.en_kn_checkpoint,
        active_settings.reader_encoder_checkpoint,
        active_settings.reader_weights,
    )
    device = resources["device"]

    question_en = _translate(resources["kn_en_runtime"], question_kn, device, active_settings)
    answer_en, score = extract_answer(
        question_en,
        context_en,
        resources["reader_model"],
        resources["tokenizer"],
        device=str(device),
    )

    answer_kn = ""
    if answer_en:
        answer_kn = _translate(resources["en_kn_runtime"], answer_en, device, active_settings)

    return {
        "question_kn": question_kn,
        "question_en": question_en,
        "context_en": context_en,
        "highlighted_context_en": highlight_answer(context_en, answer_en),
        "answer_en": answer_en,
        "answer_kn": answer_kn,
        "answer_score": "{:.2f}".format(score),
        "device": str(device),
        "kn_en_backend": active_settings.kn_en_backend,
        "en_kn_backend": active_settings.en_kn_backend,
    }
