import json
import os

import torch
import torch.nn as nn

from pretrain.mlm_final.config import ModelConfig
from pretrain.mlm_final.model import BertEncoder

class CustomBertEncoderAdapter(nn.Module):
    def __init__(self, checkpoint_dir):
        super().__init__()
        # Load config
        with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
            cfg_dict = json.load(f)
        cfg = ModelConfig(**cfg_dict)
        cfg.max_position_embeddings = 512
        self.hidden_size = cfg.hidden_size
        
        # Load encoder
        self.encoder = BertEncoder(cfg)
        
        # Load weights
        payload = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location="cpu")
        model_state = payload["model"]
        
        # The pretrain script uses `BertForMLM` which has `encoder` and `mlm`.
        # Keys are prefixed with `encoder.` or `_orig_mod.encoder.`
        final_state = {}
        for k, v in model_state.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("encoder."):
                final_state[k[len("encoder."):]] = v
                
        # Resize position embeddings
        ckpt_pos = final_state.get("embeddings.position_embeddings.weight")
        if ckpt_pos is not None and ckpt_pos.shape[0] < 512:
            new_pos = self.encoder.embeddings.position_embeddings.weight.data.clone()
            copy_len = min(ckpt_pos.shape[0], 512)
            new_pos[:copy_len] = ckpt_pos[:copy_len]
            final_state["embeddings.position_embeddings.weight"] = new_pos

        self.encoder.load_state_dict(final_state, strict=True)
        
    def encode_tokens(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # the model returns hidden states
        hidden = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return hidden
