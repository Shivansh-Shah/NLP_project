# Frontend

Run the Kannada QA Studio from the repo root:

```bash
streamlit run frontend/app.py
```

What is connected:

- PDF upload + page selection to build context
- Manual context mode for direct testing
- Kannada question -> KN->EN translation -> English QA reader -> EN->KN answer translation
- Translation model family selector with only two choices: `scratch` or `nn_transformer`
- Runtime controls (auto/cuda/cpu device and optional CPU quantization for scratch models)
- Path health checks for all model checkpoints before execution
- Cached model loading and explicit cache reset button

When `scratch` is selected, default translation checkpoints are:

- KN->EN: `translation/scratch_transformer/artifacts/kannada_to_english_scratch/checkpoints/kn_en_40m_v1_20260422/best.pt`
- EN->KN: `translation/scratch_transformer/artifacts/english_to_kannada_scratch/checkpoints/en_kn_40m_v1_20260422/best.pt`

When `nn_transformer` is selected, default translation checkpoints are:

- KN->EN: `translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt`
- EN->KN: `translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt`

Reader defaults:

- Encoder: `pretrain/mlm_final/step_100000`
- Finetuned reader weights: `finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt`
