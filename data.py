"""Module 10 (part): synthetic traffic dataset generator -> data/traffic.csv"""
import numpy as np, pandas as pd, os
from common import DATA

def generate(days=60, seed=42):
    rng = np.random.default_rng(seed)
    zones = ["Downtown", "Airport Rd", "University", "Industrial"]
    zone_base = {"Downtown": 1.0, "Airport Rd": 0.7, "University": 0.8, "Industrial": 0.6}
    rows = []
    start = pd.Timestamp("2026-01-05")
    for d in range(days):
        for h in range(24):
            ts = start + pd.Timedelta(days=d, hours=h)
            dow = ts.dayofweek
            rush = np.exp(-((h-8)**2)/4) + 1.2*np.exp(-((h-17.5)**2)/5)   # two rush hours
            base = 0.15 + 0.85*rush*(0.6 if dow >= 5 else 1.0)
            weather = rng.choice(["clear", "rain", "fog"], p=[.7, .22, .08])
            wf = {"clear": 1.0, "rain": 1.15, "fog": 1.1}[weather]
            for z in zones:
                load = np.clip(base*zone_base[z]*wf + rng.normal(0, .07), 0.02, 1.3)
                vehicles = int(load*400 + rng.normal(0, 12))
                occupancy = float(np.clip(load*0.75 + rng.normal(0, .03), 0.01, 0.99))
                speed = float(np.clip(60 - 45*load + rng.normal(0, 3), 5, 70))
                level = "Low" if load < .35 else ("Medium" if load < .7 else "High")
                rows.append((ts, z, h, dow, weather, max(vehicles, 0), round(speed, 1), round(occupancy, 3), level))
    df = pd.DataFrame(rows, columns=["timestamp","zone","hour","day_of_week","weather",
                                     "vehicle_count","avg_speed","occupancy","traffic_level"])
    df.to_csv(os.path.join(DATA, "traffic.csv"), index=False)
    return df

def load():
    p = os.path.join(DATA, "traffic.csv")
    return pd.read_csv(p, parse_dates=["timestamp"]) if os.path.exists(p) else generate()

if __name__ == "__main__":
    print(generate().head())
