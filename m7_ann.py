"""Module 7: Artificial Neural Network from scratch in NumPy (forward prop, backprop, weight updates).
(TensorFlow is optional in the brief; writing it by hand shows backpropagation explicitly.)"""
import numpy as np, pandas as pd
from data import load
LABELS = ["Low", "Medium", "High"]

def relu(z): return np.maximum(0, z)
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

class NeuralNet:
    """Input -> Hidden(ReLU) -> Hidden(ReLU) -> Output(softmax). Cross-entropy loss, mini-batch SGD."""
    def __init__(self, sizes, lr=.05, seed=0):
        r = np.random.default_rng(seed); self.lr = lr
        self.W = [r.normal(0, np.sqrt(2/a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]
    def forward(self, X):
        self.A = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.A[-1] @ W + b
            self.A.append(softmax(z) if i == len(self.W)-1 else relu(z))
        return self.A[-1]
    def backward(self, Y):
        n = Y.shape[0]; d = (self.A[-1] - Y) / n                    # dL/dz for softmax + cross-entropy
        for i in reversed(range(len(self.W))):
            gW, gb = self.A[i].T @ d, d.sum(0, keepdims=True)
            if i > 0: d = (d @ self.W[i].T) * (self.A[i] > 0)         # propagate error back through ReLU
            self.W[i] -= self.lr * gW; self.b[i] -= self.lr * gb      # weight adjustment
    def fit(self, X, y, epochs=150, batch=64, Xv=None, yv=None, seed=0):
        r = np.random.default_rng(seed); Y = np.eye(3)[y]; hist = []
        for ep in range(epochs):
            idx = r.permutation(len(X))
            for k in range(0, len(X), batch):
                j = idx[k:k+batch]; self.forward(X[j]); self.backward(Y[j])
            p = self.forward(X); loss = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
            hist.append((loss, (p.argmax(1) == y).mean(), None if Xv is None else (self.predict(Xv) == yv).mean()))
        return hist
    def predict(self, X): return self.forward(X).argmax(1)

FEATURES = ["vehicle_count", "hour", "avg_speed", "occupancy"]
def train(df=None, seed=0):
    df = load() if df is None else df
    X = df[FEATURES].values.astype(float).copy()
    X = np.c_[X[:, [0, 2, 3]], np.sin(2*np.pi*X[:, 1]/24), np.cos(2*np.pi*X[:, 1]/24)]   # cyclic time encoding
    y = df["traffic_level"].map({l: i for i, l in enumerate(LABELS)}).values
    r = np.random.default_rng(seed); idx = r.permutation(len(X)); cut = int(.75*len(X))
    mu, sd = X[idx[:cut]].mean(0), X[idx[:cut]].std(0) + 1e-9; X = (X-mu)/sd
    tr, te = idx[:cut], idx[cut:]
    net = NeuralNet([X.shape[1], 16, 8, 3], lr=.1, seed=seed)
    hist = net.fit(X[tr], y[tr], epochs=120, Xv=X[te], yv=y[te])
    acc = (net.predict(X[te]) == y[te]).mean()
    return dict(net=net, history=hist, accuracy=float(acc), mu=mu, sd=sd)

def plot_training(res, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = res["history"]; fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot([a for a, _, _ in h]); ax[0].set_title("Training loss (cross-entropy)"); ax[0].set_xlabel("epoch")
    ax[1].plot([b for _, b, _ in h], label="train"); ax[1].plot([c for _, _, c in h], label="test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

if __name__ == "__main__":
    r = train(); print("ANN test accuracy:", round(r["accuracy"], 3), "final loss:", round(r["history"][-1][0], 4))
