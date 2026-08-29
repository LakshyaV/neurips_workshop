"""Merge the regenerated E1 rows (from checkpoints) with the retrained small groups into results/E1.parquet."""
import os, glob, pandas as pd
RES = os.path.join(os.path.dirname(__file__), "..", "results")
parts = [os.path.join(RES, "E1_regen.parquet"), os.path.join(RES, "E1_n40.parquet")] + sorted(glob.glob(os.path.join(RES, "n*_small.parquet")))
dfs = [pd.read_parquet(p) for p in parts if os.path.exists(p)]
df = pd.concat(dfs, ignore_index=True)
df["group"] = df.apply(lambda r: f"E1_m{int(r.m)}_n{int(r.n)}", axis=1)
df = df.drop_duplicates(subset=["run_id", "feature_idx"], keep="last")
df.to_parquet(os.path.join(RES, "E1.parquet"), index=False)
print("E1.parquet:", len(df), "rows;", df.run_id.nunique(), "models; groups:", sorted(df.group.unique()))
