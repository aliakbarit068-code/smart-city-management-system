"""Module 10: Data collection -> preprocessing -> analysis -> visualisation -> model performance."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data import load
from common import OUT

def preprocess(df):
    d = df.drop_duplicates().copy()
    d = d[(d.vehicle_count >= 0) & d.avg_speed.between(0, 130) & d.occupancy.between(0, 1)]
    d = d.dropna(); d["is_weekend"] = d.day_of_week >= 5; return d

def summary(df):
    hourly = df.groupby("hour").vehicle_count.mean()
    return dict(total_vehicles=int(df.vehicle_count.sum()),
                avg_density=float(df.occupancy.mean()),
                peak_hours=hourly.sort_values(ascending=False).head(3).index.tolist(),
                level_counts=df.traffic_level.value_counts().to_dict(),
                busiest_zone=df.groupby("zone").vehicle_count.mean().idxmax())

def make_charts(df, outdir=OUT):
    paths = []
    hourly = df.groupby(["hour", "is_weekend"]).vehicle_count.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.5)); hourly.plot(ax=ax); ax.legend(["Weekday", "Weekend"]); ax.set_ylabel("avg vehicles")
    ax.set_title("Traffic by hour: peak hours"); p = os.path.join(outdir, "ds_peak_hours.png"); fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    piv = df.pivot_table(index="zone", columns="hour", values="occupancy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3)); im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd"); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("hour"); ax.set_title("Congestion pattern (road occupancy)"); fig.colorbar(im); p = os.path.join(outdir, "ds_heatmap.png"); fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    return paths

def model_performance():
    import m6_ml, m7_ann
    s = m6_ml.train_svm(); a = m7_ann.train()
    return pd.DataFrame([{"model": "SVM (RBF)", "test_accuracy": round(s["accuracy"], 3)},
                         {"model": "ANN (16-8, NumPy)", "test_accuracy": round(a["accuracy"], 3)}])

if __name__ == "__main__":
    df = preprocess(load()); print(summary(df)); print(make_charts(df)); print(model_performance())
