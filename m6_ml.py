"""Module 6: Supervised (SVM) and unsupervised (K-Means) learning for traffic."""
import numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from data import load

FEATURES = ["vehicle_count", "hour", "day_of_week", "weather", "occupancy"]
LABELS = ["Low", "Medium", "High"]

def prepare(df):
    X = pd.get_dummies(df[FEATURES], columns=["weather"]).astype(float)
    return X, df["traffic_level"]

def train_svm(df=None, seed=1):
    df = load() if df is None else df
    X, y = prepare(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale")).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return dict(model=model, columns=list(X.columns), accuracy=accuracy_score(yte, pred),
                confusion=confusion_matrix(yte, pred, labels=LABELS),
                report=classification_report(yte, pred, labels=LABELS, zero_division=0))

def svm_2d_demo(df=None):
    """2-feature SVM (vehicle_count vs occupancy, Low vs High) to draw hyperplane + margins like the brief's figure."""
    df = load() if df is None else df
    d = df[df.traffic_level.isin(["Low", "High"])].sample(200, random_state=0)
    X = d[["vehicle_count", "occupancy"]].values; y = (d.traffic_level == "High").astype(int).values
    sc = StandardScaler().fit(X); svm = SVC(kernel="linear", C=1).fit(sc.transform(X), y)
    return sc, svm, X, y

def kmeans_patterns(df=None, k=3):
    df = load() if df is None else df
    Z = StandardScaler().fit_transform(df[["vehicle_count", "avg_speed", "occupancy"]])
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    out = df.copy(); out["cluster"] = km.labels_
    summary = out.groupby("cluster")[["vehicle_count", "avg_speed", "occupancy"]].mean().round(2)
    summary["rows"] = out.groupby("cluster").size()
    # name clusters by mean vehicle count
    order = summary["vehicle_count"].sort_values().index.tolist()
    names = dict(zip(order, ["Low-traffic pattern", "Medium-traffic pattern", "High-traffic pattern"]))
    summary["meaning"] = [names[i] for i in summary.index]
    return out, summary

def plot_svm(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc, svm, X, y = svm_2d_demo(); Z = sc.transform(X)
    xx, yy = np.meshgrid(np.linspace(Z[:,0].min()-.5, Z[:,0].max()+.5, 200), np.linspace(Z[:,1].min()-.5, Z[:,1].max()+.5, 200))
    f = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(6, 4.5)); plt.scatter(Z[:,0], Z[:,1], c=y, cmap="coolwarm", s=18)
    plt.contour(xx, yy, f, levels=[-1, 0, 1], colors=["r", "k", "r"], linestyles=["--", "-", "--"])
    plt.scatter(*svm.support_vectors_.T, s=80, facecolors="none", edgecolors="k")
    plt.xlabel("vehicle_count (scaled)"); plt.ylabel("occupancy (scaled)"); plt.title("SVM: hyperplane, margins, support vectors")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

if __name__ == "__main__":
    r = train_svm(); print("SVM accuracy:", round(r["accuracy"], 3)); print(r["confusion"]); print(r["report"])
    print(kmeans_patterns()[1])
