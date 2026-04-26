try:
    from .train_scratch import main
except ImportError:  # pragma: no cover - direct script execution fallback
    from train_scratch import main


if __name__ == "__main__":
    main(
        default_source_lang="en",
        default_target_lang="kn",
        default_cache_dir="translation/scratch_transformer/artifacts/english_to_kannada_scratch",
        default_sp_model_name="english_to_kannada_scratch_unigram_v1",
    )
