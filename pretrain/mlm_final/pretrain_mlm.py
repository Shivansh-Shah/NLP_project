"""Compatibility wrapper for MLM pretraining.

The actual model, config, data, and training logic now live in separate files:
- config.py
- model.py
- data.py
- train.py
"""

try:
    from .config import ModelConfig
    from .data import StreamingTextDataset, collate_mlm
    from .model import BertEmbeddings, MultiHeadSelfAttention, FeedForward, TransformerBlock, BertEncoder, MLMHead, BertForMLM
    from .train import main
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import ModelConfig
    from data import StreamingTextDataset, collate_mlm
    from model import BertEmbeddings, MultiHeadSelfAttention, FeedForward, TransformerBlock, BertEncoder, MLMHead, BertForMLM
    from train import main


if __name__ == "__main__":
    main()