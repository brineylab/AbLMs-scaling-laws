#!/usr/bin/env python
"""Zero-shot pseudo-perplexity scoring of paired antibody sequences.

For each sequence we mask each residue position one at a time (excluding the
leading <cls>, trailing <eos>, and the two interior <cls><cls> separator
tokens), read the model's log-prob of the true residue, average over positions,
and report pseudo_ppl = exp(-mean log-prob). Lower = the model finds the
sequence more "natural". Correlating this with experimental fitness (per assay
campaign) is the FLAb-style zero-shot evaluation.

The masking for one sequence is batched (one forward over all masked variants),
chunked to fit GPU memory for the larger models.

Usage:
    python 02_zero-shot_pseudo-perplexity.py --model 650M_full --dataset binding
"""
import os, math, argparse, time
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import EsmForMaskedLM, AutoTokenizer
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model key in models.yaml")
    p.add_argument("--dataset", required=True, help="binding|expression|tm")
    p.add_argument("--config", default=os.path.join(HERE, "models.yaml"))
    p.add_argument("--chunk", type=int, default=0,
                   help="masked-position sub-batch size (0=auto by model size)")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--data_dir", default="data",
                   help="directory holding flab2_<property>_full.csv, relative to this script")
    p.add_argument("--max_tokens", type=int, default=1000)
    p.add_argument("--nshards", type=int, default=1, help="split dataset into N shards")
    p.add_argument("--shard", type=int, default=0, help="which shard (0..N-1)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def auto_chunk(hidden):
    # how many masked copies of one sequence go through the model at once; sized for a 46 GB GPU.
    # A CUDA out-of-memory error halves the sub-batch and retries, so these can be generous.
    if hidden >= 1280:   # 650M
        return 128
    if hidden >= 960:    # 350M
        return 256
    if hidden >= 640:    # 150M
        return 384
    return 512           # 8M / 35M


@torch.no_grad()
def _forward_positions(input_ids, pos, model, mask_id):
    idx = torch.arange(len(pos), device=input_ids.device)
    batch = input_ids.repeat(len(pos), 1).clone()
    batch[idx, pos] = mask_id
    logits = model(batch).logits                            # (n, L, V)
    lp = torch.log_softmax(logits.float(), dim=-1)
    wt = input_ids[0, pos]                                  # true residues
    return lp[idx, pos, wt]


@torch.no_grad()
def pseudo_ppl(input_ids, score_pos, model, mask_id, chunk):
    """input_ids: (1, L) on device. score_pos: list of positions to mask.
    Batches masked variants; on CUDA OOM the sub-batch is halved and retried."""
    pos_t = torch.as_tensor(score_pos, device=input_ids.device)
    logps = []
    s = 0
    while s < len(pos_t):
        c = min(chunk, len(pos_t) - s)
        while True:
            try:
                logps.append(_forward_positions(input_ids, pos_t[s:s + c], model, mask_id))
                s += c
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if c == 1:
                    raise
                c = max(1, c // 2)
    ll = torch.cat(logps).mean().item()
    return math.exp(-ll)


def main():
    a = parse()
    torch.set_num_threads(8)   # keep the CPU from being oversubscribed by parallel jobs
    cfg = yaml.safe_load(open(a.config))
    m = cfg["models"][a.model]
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    mask_id = tok.mask_token_id

    ds_path = os.path.join(HERE, a.data_dir, f"flab2_{a.dataset}_full.csv")
    data = pd.read_csv(ds_path)
    # strided sharding: shard k processes rows k, k+N, k+2N, ... (balanced across
    # source_files); the ORIGINAL row index is kept as seq_id so shards merge.
    orig_idx = list(range(a.shard, len(data), a.nshards)) if a.nshards > 1 else list(range(len(data)))
    data = data.iloc[orig_idx].reset_index(drop=True)

    model = EsmForMaskedLM.from_pretrained(m["checkpoint"])
    if a.fp16:
        model = model.half()
    model = model.to("cuda").eval()
    chunk = a.chunk or auto_chunk(m["hidden_size"])

    tag = f"_shard{a.shard}of{a.nshards}" if a.nshards > 1 else ""
    out = a.out or os.path.join(HERE, "results", "ppl", a.dataset,
                                f"{a.model}{tag}_pseudo-ppl.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    t0 = time.time()
    for i, r in enumerate(tqdm(data.itertuples(index=False), total=len(data),
                               desc=f"{a.model}/{a.dataset}")):
        ids = tok.encode(r.sequence, return_tensors="pt").to("cuda")
        L = ids.shape[1]
        if L > a.max_tokens:
            rows.append(dict(seq_id=orig_idx[i], source_file=r.source_file,
                             fitness=r.fitness, pseudo_ppl=np.nan, n_pos=0))
            continue
        idv = ids[0]
        # positions to skip: bos (idx 0), eos (last, id==2), interior <cls> (id==0)
        is_zero = (idv == 0)
        is_eos = (idv == tok.eos_token_id)
        skip = torch.zeros(L, dtype=torch.bool, device=ids.device)
        skip[0] = True
        skip |= is_zero | is_eos
        score_pos = torch.nonzero(~skip, as_tuple=False).flatten().tolist()
        ppl = pseudo_ppl(ids, score_pos, model, mask_id, chunk)
        rows.append(dict(seq_id=orig_idx[i], source_file=r.source_file,
                         fitness=r.fitness, pseudo_ppl=ppl, n_pos=len(score_pos)))

    df = pd.DataFrame(rows)
    df.insert(0, "model", a.model)
    df.insert(1, "property", a.dataset)
    df.to_csv(out, index=False)
    print(f"[{a.model}/{a.dataset}] {len(df)} seqs, {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
