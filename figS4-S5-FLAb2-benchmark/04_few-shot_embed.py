#!/usr/bin/env python
"""Step 1 of the FLAb2 few-shot evaluation: frozen embeddings of the 15 models.

Same inputs as the zero-shot evaluation: the same checkpoints (models.yaml) and the same sequence files
(data/flab2_<property>_full.csv, heavy<cls><cls>light), so a dataset here is a dataset of the zero-shot
table. Kept: every dataset with at least 100 antibodies (MIN_N), all of its rows.

One forward pass of the frozen model in half precision, last hidden state, averaged over the residue
positions only (not <cls>, <eos>, padding or the two <cls> tokens between the chains).

Writes (once)  data/fewshot_rows.csv       property, source_file, seq_id, fitness, seq_index
               data/fewshot_sequences.csv  seq_index, sequence   (each distinct sequence once)
and per model  results/embeddings/<model>.npy   float16, (distinct sequences, hidden size)
               (the model runs in half precision, so float16 stores its output without loss)

Usage: python 04_few-shot_embed.py --gpu 0 --models 8M_full,8M_half      (default: all 15)
"""
import argparse, os, time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_N = 100
PROPS = ["binding", "polyreactivity", "aggregation", "thermostability", "pharmacokinetics", "expression", "immunogenicity"]


def build_rows():
    rows_f, seq_f = os.path.join(HERE, "data", "fewshot_rows.csv"), os.path.join(HERE, "data", "fewshot_sequences.csv")
    if os.path.exists(rows_f) and os.path.exists(seq_f):
        return pd.read_csv(rows_f), pd.read_csv(seq_f)
    frames = []
    for prop in PROPS:
        d = pd.read_csv(os.path.join(HERE, "data", f"flab2_{prop}_full.csv")); d["seq_id"] = np.arange(len(d)); d["property"] = prop
        frames.append(d[["property", "source_file", "seq_id", "fitness", "sequence"]])
    d = pd.concat(frames, ignore_index=True)
    assert d.groupby(["property", "source_file"]).ngroups == 141 and len(d) == 563844          # as the zero-shot table
    n = d.groupby(["property", "source_file"]).sequence.transform("size")
    d = d[n >= MIN_N].reset_index(drop=True)
    seqs = pd.DataFrame({"sequence": pd.unique(d.sequence)}); seqs.insert(0, "seq_index", np.arange(len(seqs)))
    d["seq_index"] = d.sequence.map(dict(zip(seqs.sequence, seqs.seq_index)))
    d.drop(columns="sequence").to_csv(rows_f, index=False); seqs.to_csv(seq_f, index=False)
    return d.drop(columns="sequence"), seqs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="0"); ap.add_argument("--models", default="")
    ap.add_argument("--batch", type=int, default=64); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    rows, seqs = build_rows()
    print(f"rows {len(rows):,} in {rows.groupby(['property', 'source_file']).ngroups} datasets; distinct sequences {len(seqs):,}", flush=True)
    import torch
    from transformers import AutoTokenizer, EsmModel
    cfg = yaml.safe_load(open(os.path.join(HERE, "models.yaml")))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    cls_id, eos_id, pad_id = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id
    order = np.argsort(seqs.sequence.str.len().values, kind="stable"); texts = seqs.sequence.values[order]
    for name in ([m for m in a.models.split(",") if m] or list(cfg["models"])):
        out = os.path.join(HERE, "results", "embeddings", f"{name}.npy")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            print("exists:", out, flush=True); continue
        m = cfg["models"][name]; t0 = time.time()
        model = EsmModel.from_pretrained(m["checkpoint"], add_pooling_layer=False).half().to("cuda").eval()
        emb = np.empty((len(seqs), m["hidden_size"]), dtype=np.float16)
        with torch.no_grad():
            for s in range(0, len(texts), a.batch):
                enc = tok(list(texts[s:s + a.batch]), return_tensors="pt", padding=True).to("cuda")
                ids, am = enc["input_ids"], enc["attention_mask"]
                hs = model(input_ids=ids, attention_mask=am).last_hidden_state
                v = ((ids != cls_id) & (ids != eos_id) & (ids != pad_id) & (am == 1)).unsqueeze(-1).to(hs.dtype)
                emb[order[s:s + a.batch]] = ((hs * v).sum(1) / v.sum(1).clamp(min=1)).cpu().numpy()
        assert np.isfinite(emb).all()
        np.save(out + ".tmp.npy", emb); os.replace(out + ".tmp.npy", out)
        print(f"{name}: {emb.shape} from {m['checkpoint']} in {time.time() - t0:.0f} s", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
