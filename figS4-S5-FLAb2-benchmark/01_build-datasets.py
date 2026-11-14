#!/usr/bin/env python
"""Build per-property FLAb2 datasets for the paired scaling-law AbLMs.

FLAb2 (Graylab/FLAb v1.0.0) has 165 dataset files across 7 properties, with
heterogeneous schemas (varied column names, some .csv.zip, some with a junk
first row, some without a 'fitness' column, some VHH/scFv). Our models are
PAIRED, so we keep only datasets that expose BOTH a heavy and a light chain.

Per dataset we:
  - locate <category>/<filename> (metadata filename is a basename),
  - read (unzipping; picking the header row that exposes heavy/light),
  - detect the heavy & light columns (name contains 'heavy'/'light'),
  - detect the fitness column ('fitness' if present, else the last numeric
    non-chain column = the assay value),
  - drop rows with empty/oversized chains,
  - build sequence = heavy + '<cls><cls>' + light.

Outputs <out_dir>/flab2_<property>_full.csv (default out_dir: data/) with
heavy,light,fitness,source_file,sequence and prints an inventory (kept / skipped).

Usage:
    python 01_build-datasets.py --flab_repo /path/to/FLAb
"""
import os, io, zipfile, argparse, warnings
warnings.filterwarnings("ignore")
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "FLAb_repo", "data")   # default of --flab_repo, + /data
OUT = os.path.join(HERE, "data")
SEP = "<cls><cls>"                               # the separator these models were pretrained with
MAXLEN = 400        # drop chains longer than this (guards scFv/junk)
PROPS = ["binding", "expression", "thermostability", "aggregation",
         "pharmacokinetics", "polyreactivity", "immunogenicity"]
EXTENSIONS = (".csv.zip", ".csv")   # every FLAb2 data file ends with one of these
# Three binding files whose names contain dots keep the shortened name they were given when we ran
# this, so that the dataset names here are the ones behind S4 Fig and S5 Fig. Each stands for one
# file. Full file name without its extension -> name used:
SHORT_NAMES = {
    "shanker2024unsupervised_SA58-BA.1_IC50": "shanker2024unsupervised_SA58-BA",
    "shanker2024unsupervised_SA58-BQ.1.1_IC50": "shanker2024unsupervised_SA58-BQ.1",
    "shanker2024unsupervised_SA58-XBB.1.5_Kd": "shanker2024unsupervised_SA58-XBB.1",
}


def dataset_name(filename):
    """source_file of a FLAb2 file: the file name without its extension (.csv or .csv.zip).
    Dots inside the name are part of the name (see SHORT_NAMES)."""
    for ext in EXTENSIONS:
        if filename.endswith(ext):
            stem = filename[:-len(ext)]
            return SHORT_NAMES.get(stem, stem)
    return os.path.splitext(filename)[0]    # another extension: cut that one extension only


def read_any(path):
    """Read csv or csv.zip, choosing the header row that exposes heavy/light."""
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw = zf.read(inner)
        for hdr in (0, 1, 2):
            try:
                df = pd.read_csv(io.BytesIO(raw), header=hdr, encoding="utf-8-sig",
                                 low_memory=False)
                if any("heavy" in str(c).lower() for c in df.columns) and \
                   any("light" in str(c).lower() for c in df.columns):
                    return df
            except Exception:
                continue
        return None
    for hdr in (0, 1):
        try:
            df = pd.read_csv(path, header=hdr, encoding="utf-8-sig", low_memory=False)
            if any("heavy" in str(c).lower() for c in df.columns) and \
               any("light" in str(c).lower() for c in df.columns):
                return df
        except Exception:
            continue
    return None


def pick_col(cols, key):
    exact = [c for c in cols if str(c).lower() == key]
    if exact:
        return exact[0]
    part = [c for c in cols if key in str(c).lower()]
    return part[0] if part else None


def pick_fitness(df, hcol, lcol):
    if "fitness" in [str(c).lower() for c in df.columns]:
        return [c for c in df.columns if str(c).lower() == "fitness"][0]
    # else: last numeric column that isn't a chain / obvious id/mutation column
    bad = {hcol, lcol}
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("heavy", "light", "_mut", "mutation", "design",
                                 "antibody", "name", "species", "sex", "id")):
            bad.add(c)
    num = [c for c in df.columns if c not in bad and
           pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    return num[-1] if num else None


def build(prop, meta, out_dir):
    files = meta[meta.category == prop]
    frames, log = [], []
    names = {}          # source_file name -> file, so that two files can never share one name
    for _, r in files.iterrows():
        path = os.path.join(REPO, prop, r["filename"])
        if not os.path.exists(path):
            log.append((r["filename"], "missing", 0)); continue
        df = read_any(path)
        if df is None:
            log.append((r["filename"], "no-heavy/light", 0)); continue
        hcol = pick_col(df.columns, "heavy"); lcol = pick_col(df.columns, "light")
        fcol = pick_fitness(df, hcol, lcol)
        if not (hcol and lcol and fcol):
            log.append((r["filename"], "no-fitness-col", 0)); continue
        d = df[[hcol, lcol, fcol]].copy()
        d.columns = ["heavy", "light", "fitness"]
        d["heavy"] = d["heavy"].astype(str).str.strip().str.upper()
        d["light"] = d["light"].astype(str).str.strip().str.upper()
        d["fitness"] = pd.to_numeric(d["fitness"], errors="coerce")
        d = d.dropna(subset=["heavy", "light", "fitness"])
        d = d[(d.heavy.str.fullmatch(r"[A-Z]+")) & (d.light.str.fullmatch(r"[A-Z]+"))]
        d = d[(d.heavy.str.len().between(50, MAXLEN)) & (d.light.str.len().between(50, MAXLEN))]
        d = d.drop_duplicates(subset=["heavy", "light", "fitness"])
        if len(d) < 10:
            log.append((r["filename"], f"too-few({len(d)})", len(d))); continue
        note = "kept"
        name = dataset_name(r["filename"])
        if name in names:
            raise SystemExit(f"{prop}: {r['filename']} and {names[name]} would both get the "
                             f"source_file name {name}")
        names[name] = r["filename"]
        d["source_file"] = name
        frames.append(d); log.append((r["filename"], note, len(d)))
    if not frames:
        print(f"\n### {prop}: NO usable datasets"); return None
    full = pd.concat(frames, ignore_index=True)
    full["sequence"] = full["heavy"] + SEP + full["light"]
    out = os.path.join(out_dir, f"flab2_{prop}_full.csv")
    full.to_csv(out, index=False)
    kept = sum(1 for _, s, _ in log if s.startswith("kept"))
    print(f"\n### {prop}: {len(full)} seqs from {kept} datasets -> {os.path.basename(out)}")
    for fn, s, n in sorted(log, key=lambda x: -x[2])[:6]:
        print(f"    {n:>6}  {s:<16} {fn}")
    skipped = [x for x in log if not x[1].startswith("kept")]
    if skipped:
        print(f"    (skipped {len(skipped)}: " +
              ", ".join(f"{fn}[{s}]" for fn, s, _ in skipped[:4]) +
              (" ..." if len(skipped) > 4 else "") + ")")
    return full


def main():
    global REPO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flab_repo", default=os.path.dirname(REPO),
                    help="clone of https://github.com/Graylab/FLAb (default: FLAb_repo/ here)")
    ap.add_argument("--out_dir", default=OUT,
                    help="directory for the 7 flab2_<property>_full.csv (default: data/)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing output files")
    a = ap.parse_args()
    REPO = os.path.join(a.flab_repo, "data")
    present = [f for f in os.listdir(a.out_dir) if f.startswith("flab2_") and f.endswith("_full.csv")] \
        if os.path.isdir(a.out_dir) else []
    if present and not a.overwrite:
        ap.error(f"{a.out_dir} already holds {len(present)} dataset file(s); use --out_dir <other "
                 "folder>, or --overwrite to replace them")
    os.makedirs(a.out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(REPO, "flab_metadata.csv"))
    total = 0
    for p in PROPS:
        f = build(p, meta, a.out_dir)
        if f is not None:
            total += len(f)
    print(f"\n=== TOTAL: {total} sequences across {len(PROPS)} properties ===")


if __name__ == "__main__":
    main()
