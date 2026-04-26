# NLP Project

This repository contains a complete end-to-end NLP stack built around Kannada-English workflows:

- MLM pretraining (custom BERT-style encoder)
- Extractive QA finetuning on SQuAD
- Bidirectional machine translation (Kannada <-> English)
- Streamlit frontend that connects translation + QA in one app

## Links

- GitHub Repository: https://github.com/Shivansh-Shah/NLP_project
- Model Checkpoints (Google Drive): https://drive.google.com/drive/folders/1DGeQDnoSRhhQB4cTV6hOzUlvYhXvDEQI?usp=sharing

## Project Structure

```text
pretrain/mlm_final/                    # MLM config/model/data/train/test scripts + checkpoint metadata
finetuning/finetunningmyown/           # QA reader train/eval/demo pipeline
translation/english_to_kannada/        # nn.Transformer training + generic translate runner
translation/kannada_to_english/        # KN->EN model artifacts/tokenizers/checkpoint history
translation/scratch_transformer/        # from-scratch Transformer training/inference (no nn.Transformer blocks)
frontend/                              # Streamlit app + connected runtime pipeline
artifacts/                             # legacy tokenizer/model artifact paths used by compatibility logic
```

## Important Notes

- artifacts/mlm_pretrained.pt is the older pretrained artifact path referenced in earlier setups.
- pretrain/mlm_final/step_100000/checkpoint.pt is the main encoder checkpoint used by QA scripts.
- Large model checkpoint files are intentionally excluded from Git and should be downloaded from Drive.
- Many scripts have fallback path-resolution logic, so relative repo paths are the expected default.

### Drive-Only Checkpoints (Not In GitHub)

The following files are available in the Google Drive link above and are not included in this GitHub repository:

- finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt (https://drive.google.com/file/d/1kGKYCUpZJbVLJPWi0NhfkyFFWu4IxJej/view?usp=share_link)
- pretrain/mlm_final/step_100000/checkpoint.pt (https://drive.google.com/file/d/1zxw1l4whPy8Ooz9n2RKQZHhFd2ANGz09/view?usp=sharing)
- translation/scratch_transformer/artifacts/english_to_kannada_scratch/checkpoints/full_train_20260422_gpuonly_seq/best.pt (https://drive.google.com/file/d/1D_m9d-rNDRmxaXrpwDwVRvwnAiFppRud/view?usp=sharing)
- translation/scratch_transformer/artifacts/kannada_to_english_scratch/checkpoints/full_train_20260422_gpuonly_seq/best.pt (https://drive.google.com/file/d/1WC0t-6zgyHV6R-HqyXei-64FTAt4lNZR/view?usp=share_link)
- translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt (https://drive.google.com/file/d/1WoOThm5B7pAR8DxB1MFpjY_KUbZm4r94/view?usp=sharing)
- translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt (https://drive.google.com/file/d/1w9Vj9KJShxDWPW9ON6fzcgTVF4Hww4AM/view?usp=share_link)

After downloading, place them at the same relative paths in this repository.

## Environment Setup

Use Python 3.10+ recommended.

```bash
pip install -U torch transformers datasets sentencepiece sacrebleu tqdm streamlit pypdf
```

Optional package for Kannada romanization output:

```bash
pip install indic-transliteration
```

## Run Guide

### 1) MLM Pretraining

Quick model test using an existing checkpoint:

```bash
python -m pretrain.mlm_final.test_mlm
```

Full training entrypoint:

```bash
python -m pretrain.mlm_final.train --out_dir pretrain/mlm_final --max_steps 100000
```

### 2) QA Finetuning and Evaluation

Train reader on local SQuAD files:

```bash
python -m finetuning.finetunningmyown.train_reader \
	--train-json squad/train-v1.1.json \
	--dev-json squad/dev-v1.1.json \
	--encoder-checkpoint pretrain/mlm_final/step_100000
```

Evaluate reader:

```bash
python -m finetuning.finetunningmyown.eval_squad \
	--dev-json squad/dev-v1.1.json \
	--weights finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt
```

Run full QA demo (Kannada question -> English QA -> Kannada answer):

```bash
python -m finetuning.finetunningmyown.full_pipeline_demo \
	--context "Apollo 11 was the first mission to land humans on the Moon." \
	--question-kn "ಅಪೋಲೊ 11 ಏನು?"
```

### 3) Translation (nn.Transformer checkpoints)

Note: the generic translation runner lives in translation/english_to_kannada/translate.py and can load either direction depending on checkpoint metadata.

Kannada -> English:

```bash
python -m translation.english_to_kannada.translate \
	--checkpoint translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt \
	--text "ನಿಮ್ಮ ಪಠ್ಯ ಇಲ್ಲಿ"
```

English -> Kannada:

```bash
python -m translation.english_to_kannada.translate \
	--checkpoint translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt \
	--text "Your text here"
```

### 4) Translation (Scratch Transformer)

Train EN -> KN scratch model:

```bash
python -m translation.scratch_transformer.train_en_kn_scratch \
	--save-dataset-to-disk \
	--max-samples 600000 \
	--epochs 6
```

Train KN -> EN scratch model:

```bash
python -m translation.scratch_transformer.train_kn_en_scratch \
	--save-dataset-to-disk \
	--max-samples 600000 \
	--epochs 6
```

Inference with scratch model:

```bash
python -m translation.scratch_transformer.translate_scratch \
	--checkpoint translation/scratch_transformer/artifacts/kannada_to_english_scratch/checkpoints/kn_en_40m_v1_20260422/best.pt \
	--text "ನಿಮ್ಮ ಪಠ್ಯ ಇಲ್ಲಿ"
```

### 5) Frontend App

Run the Streamlit interface from repo root:

```bash
streamlit run frontend/app.py
```

The app supports:

- PDF upload + page selection for context
- Manual context mode
- Kannada question input
- Model family toggle: scratch or nn_transformer
- Path health checks for checkpoints
- Cached model loading and manual cache reset

## Default Runtime Paths Used by Code

- Reader encoder checkpoint: pretrain/mlm_final/step_100000
- Reader finetuned weights: finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt
- KN->EN nn.Transformer: translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt
- EN->KN nn.Transformer: translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt
- KN->EN scratch: translation/scratch_transformer/artifacts/kannada_to_english_scratch/checkpoints/kn_en_40m_v1_20260422/best.pt
- EN->KN scratch: translation/scratch_transformer/artifacts/english_to_kannada_scratch/checkpoints/en_kn_40m_v1_20260422/best.pt

## Data Sources

- MLM pretraining stream: wikimedia/wikipedia (20231101.en) via Hugging Face datasets
- Translation training: ai4bharat/samanantar
- QA finetuning/evaluation: SQuAD v1.1 (local JSON files expected)

## Troubleshooting

- If a checkpoint path is missing, download model files from the Drive link and place them at the paths above.
- If CUDA is unavailable, most scripts fall back to CPU; QA trainer currently asserts GPU availability for training.
- If tokenizer file resolution fails, verify SentencePiece model files are present in the expected tokenizer folders.
