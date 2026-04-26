import torch
from transformers import AutoTokenizer
import json
import os

from .config import ModelConfig
from .model import BertForMLM

def test_mask(sentence, checkpoint_dir="pretrain/mlm_final/step_100000"):
    # Load config
    with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)
    
    # Load model
    model = BertForMLM(cfg)
    payload = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location="cpu")
    # Some layers might have a prefix depending on compilation, so we use strict=False or clean up keys
    model_state = payload["model"]
    # If compiled, keys might have '_orig_mod.' prefix
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # Preprocess
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))
    
    # Find mask token index
    mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    if len(mask_token_index) == 0:
        print(f"No {tokenizer.mask_token} token found in sentence!")
        return
        
    with torch.no_grad():
        logits, _ = model(input_ids, token_type_ids, attention_mask)
        
    # Get top 5 predictions for each mask token
    print(f"\nSentence: {sentence}")
    for idx in mask_token_index:
        mask_logits = logits[0, idx, :]
        top_5_tokens = torch.topk(mask_logits, 5, dim=-1).indices.tolist()
        
        print("Predictions for [MASK]:")
        for i, token in enumerate(top_5_tokens, 1):
            decoded_word = tokenizer.decode([token])
            print(f"  {i}. {decoded_word}")

if __name__ == "__main__":
    sentences = [
        "The capital of France is [MASK].",
        "I want to [MASK] a new language.",
        "The quick brown [MASK] jumps over the lazy dog.",
        "She works as a software [MASK] at a tech company."
    ]
    for s in sentences:
        test_mask(s)
