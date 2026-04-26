# NLP Project Layout

This workspace is now organized by stage:

- `pretrain/mlm_final/` - MLM pretraining code and checkpoints
- `finetuning/finetunningmyown/` - reader finetuning, evaluation, and demo scripts
- `translation/kannada_to_english/` - Kannada -> English translation model and checkpoints
- `translation/english_to_kannada/` - English -> Kannada translation model and checkpoints

Important distinction:

- `artifacts/mlm_pretrained.pt` is the older pretrained MLM artifact
- `pretrain/mlm_final/step_100000/checkpoint.pt` is the checkpoint used for the reader/QA encoder
Canonical translation model paths:

- Kannada -> English: `translation/kannada_to_english/kannada_to_english_v2`
- English -> Kannada: `translation/english_to_kannada`

Recommended entry points:

- Pretrain: `python -m pretrain.mlm_final.test_mlm`
- Finetune evaluation: `python -m finetuning.finetunningmyown.eval_squad`
- Full QA pipeline: `python -m finetuning.finetunningmyown.full_pipeline_demo`
- Kannada -> English translation: `python -m translation.english_to_kannada.translate --checkpoint translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt --text "..."`
- English -> Kannada translation: `python -m translation.english_to_kannada.translate --checkpoint translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt --text "..."`

Default model paths used by the scripts:

- MLM checkpoint: `pretrain/mlm_final/step_100000`
- Finetuned reader weights: `finetuning/finetunningmyown/artifacts/finetunningmyown/best.pt`
- KN -> EN checkpoint: `translation/kannada_to_english/kannada_to_english_v2/checkpoints/best.pt`
- EN -> KN checkpoint: `translation/english_to_kannada/artifacts/english_to_kannada/checkpoints/10m_long_v2/best.pt`

Rollback helper:

- `restore_legacy_aliases.sh` recreates the removed compatibility symlinks if you need the old names back.