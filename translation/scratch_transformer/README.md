# Scratch Transformer EN<->KN Translation

This module is fully isolated from the existing translator files.

It mirrors the existing training flow but uses a from-scratch Transformer implementation:

- No `nn.Transformer`, `nn.TransformerEncoder`, or `nn.TransformerDecoder`
- Manual multi-head attention with Q/K/V projections and scaled dot-product attention
- Manual encoder and decoder blocks
- Shared SentencePiece tokenizer flow and checkpointing

## Files

- `model_scratch.py`: custom Transformer blocks and model
- `train_scratch.py`: common trainer
- `train_en_kn_scratch.py`: EN -> KN training entrypoint
- `train_kn_en_scratch.py`: KN -> EN training entrypoint
- `translate_scratch.py`: greedy inference from checkpoint

## Example training

EN -> KN:

```bash
python translation/scratch_transformer/train_en_kn_scratch.py \
  --save-dataset-to-disk \
  --max-samples 600000 \
  --epochs 6
```

KN -> EN:

```bash
python translation/scratch_transformer/train_kn_en_scratch.py \
  --save-dataset-to-disk \
  --max-samples 600000 \
  --epochs 6
```

## Quick smoke run

Use this first to verify setup:

```bash
python translation/scratch_transformer/train_en_kn_scratch.py \
  --max-samples 2000 \
  --epochs 1 \
  --max-train-steps 20
```
