import numpy as np
import sacrebleu
import torch


def noam_lr_lambda(step, warmup_steps):
    step = max(1, step)
    return min(step ** (-0.5), step * (warmup_steps ** -1.5)) * (warmup_steps ** 0.5)


def evaluate(model, loader, criterion, device, use_amp, pad_id, amp_autocast_fn):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            with amp_autocast_fn(device, use_amp):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            n_tokens = int(tgt_out.ne(pad_id).sum().item())
            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def greedy_decode(model, sp, text, device, max_seq_len, max_gen_len, bos_id, eos_id):
    model.eval()

    src_ids = [bos_id] + sp.encode(text, out_type=int)[: max_seq_len - 2] + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    memory, src_mask = base_model.encode(src)

    tgt_ids = [bos_id]
    for _ in range(max_gen_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        next_logits = base_model.decode_step(tgt, memory, src_mask)
        next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == eos_id:
            break
        tgt_ids.append(next_id)

    return sp.decode(tgt_ids[1:])


def evaluate_bleu(model, sp, val_text_ds, device, max_seq_len, max_samples, bos_id, eos_id):
    sample_size = min(len(val_text_ds), max_samples)
    if sample_size <= 0:
        return 0.0

    indices = np.random.choice(len(val_text_ds), size=sample_size, replace=False)
    hypotheses = []
    references = []

    for idx in indices:
        row = val_text_ds[int(idx)]
        pred = greedy_decode(
            model=model,
            sp=sp,
            text=row["src"],
            device=device,
            max_seq_len=max_seq_len,
            max_gen_len=100,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        hypotheses.append(pred)
        references.append(row["tgt"])

    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
