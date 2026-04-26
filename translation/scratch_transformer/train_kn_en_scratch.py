try:
    from .train_scratch import main
except ImportError:  # pragma: no cover - direct script execution fallback
    from train_scratch import main


if __name__ == "__main__":
    main(
        default_source_lang="kn",
        default_target_lang="en",
        default_cache_dir="translation/scratch_transformer/artifacts/kannada_to_english_scratch",
        default_sp_model_name="kannada_to_english_scratch_unigram_v1",
    )
