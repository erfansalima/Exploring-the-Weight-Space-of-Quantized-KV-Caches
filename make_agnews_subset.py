"""Builds agnews_subset.jsonl (seed 1234): 4,000 balanced train examples + the full official test set (7,600).
Source: AG News CSVs (train.csv / test.csv, columns: class 1-4, title, description), e.g. from
https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv"""
import sys, json
import numpy as np, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "ag_news_csv/"
names = ["World", "Sports", "Business", "Sci/Tech"]
def load(f):
    df = pd.read_csv(d + f, header=None, names=["y", "title", "desc"])
    clean = lambda s: " ".join(str(s).replace("\\", " ").split())
    df["title"], df["desc"], df["y"] = df.title.map(clean), df.desc.map(clean), df.y - 1
    return df
tr, te = load("train.csv"), load("test.csv")
rng = np.random.default_rng(1234)
idx = np.concatenate([rng.choice(np.where(tr.y == c)[0], 1000, replace=False) for c in range(4)]); rng.shuffle(idx)
with open("agnews_subset.jsonl", "w") as f:
    for split, df, ii in (("train", tr, idx), ("test", te, np.arange(len(te)))):
        for i in ii:
            r = df.iloc[i]
            f.write(json.dumps(dict(split=split, src_row=int(i), label=int(r.y), label_name=names[r.y], title=r.title, text=r.desc)) + "\n")
