# EN->KN Translation (Samanantar, Method 2)

This folder contains a full translation pipeline using Hugging Face datasets:

- Downloads `ai4bharat/samanantar` with `en-kn`
- Saves dataset locally on remote machine (optional)
- Trains a SentencePiece tokenizer from streamed training text
- Trains a Transformer NMT model with bucketed token batches, AMP, and checkpointing
- Runs greedy decoding for inference

## Install

Use your GPU Python environment:

```bash
/home/user29/miniforge3/envs/legalgpu/bin/python -m pip install -U datasets sentencepiece sacrebleu pyarrow pandas
```

## Train

From workspace root (`/home/user29/NLP`):

```bash
/home/user29/miniforge3/envs/legalgpu/bin/python translation/english_to_kannada/train_en_kn.py \
  --save-dataset-to-disk \
  --cache-dir artifacts/translation_en_kn \
  --max-samples 600000 \
  --epochs 6
```

Notes:

- Use `--use-saved-dataset` in later runs to avoid re-download.
- This trainer uses the fixed 10M translation config.
- Increase `--max-tokens-per-batch` if GPU memory allows.

## Translate

```bash
/home/user29/miniforge3/envs/legalgpu/bin/python language_translation/translate.py \
  --checkpoint artifacts/translation_en_kn/checkpoints/best.pt \
  --text "Machine translation helps break language barriers." \
  --device cuda
```

## Outputs

- Best checkpoint: `artifacts/translation_en_kn/checkpoints/best.pt`
- Training log history: `artifacts/translation_en_kn/checkpoints/train_history.json`
- SentencePiece tokenizer: `artifacts/translation_en_kn/tokenizer/*.model`