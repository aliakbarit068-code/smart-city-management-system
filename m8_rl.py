"""Module 8: Q-learning traffic-signal agent vs. fixed-time control on a single 4-way junction."""
import numpy as np
import random

class Junction:
    """State: (NS queue bucket, EW queue bucket, current green, emergency side or none). Action: 0 keep, 1 switch.
    Each step = 5 s. Cars arrive randomly; green direction discharges up to 3 cars."""
    def __init__(self, seed=0): self.r = random.Random(seed); self.reset()
    def reset(self):
        self.q = [0, 0]; self.green = 0; self.emerg = None; self.t = 0; self.since_switch = 0
        return self.state()
    def bucket(self, n): return 0 if n < 4 else (1 if n < 9 else 2)
    def state(self): return (self.bucket(self.q[0]), self.bucket(self.q[1]), self.green, -1 if self.emerg is None else self.emerg)
    def step(self, action):
        reward = 0
        if action == 1 and self.since_switch >= 2:           # min green of 2 steps
            self.green ^= 1; self.since_switch = 0; reward -= 5   # penalty: unnecessary switching / lost time
        else: self.since_switch += 1
        if self.emerg is None and self.r.random() < .03: self.emerg = self.r.randint(0, 1)
        for d in (0, 1):
            if self.r.random() < (.55 if d == 0 else .45): self.q[d] += self.r.choice([1, 1, 2])
        served = min(self.q[self.green], 3); self.q[self.green] -= served
        waiting = sum(self.q); reward += 10 if served >= 2 else 0; reward -= waiting*0.5
        if self.q[1-self.green] > 0 and self.q[self.green] == 0: reward -= 5      # unnecessary red/empty green
        if self.emerg is not None:
            if self.emerg == self.green: self.emerg = None; reward += 20       # emergency cleared
            else: reward -= 20                                                  # emergency delayed
        self.t += 1
        return self.state(), reward, self.t >= 200

def fixed_time(state, t, period=6): return 1 if t % period == period-1 else 0

def train(episodes=600, alpha=.15, gamma=.9, eps0=1.0, seed=0):
    env = Junction(seed); Q = {}; rewards = []; rnd = random.Random(seed)
    def q(s): return Q.setdefault(s, [0.0, 0.0])
    for ep in range(episodes):
        s, tot, done = env.reset(), 0, False
        eps = max(.05, eps0 * (1 - ep/(episodes*.8)))
        while not done:
            a = rnd.randint(0, 1) if rnd.random() < eps else int(np.argmax(q(s)))
            s2, r, done = env.step(a)
            q(s)[a] += alpha*(r + gamma*max(q(s2)) - q(s)[a]); s = s2; tot += r
        rewards.append(tot)
    return Q, rewards

def evaluate(policy, episodes=50, seed=123):
    env = Junction(seed); res = []
    for _ in range(episodes):
        s, tot, done, wait = env.reset(), 0, False, 0
        while not done:
            s, r, done = env.step(policy(s, env.t)); tot += r; wait += sum(env.q)
        res.append((tot, wait/200))
    a = np.array(res); return a[:, 0].mean(), a[:, 1].mean()

def compare(Q=None):
    if Q is None: Q, _ = train()
    learned = lambda s, t: int(np.argmax(Q.get(s, [0, 0])))
    fr, fw = evaluate(lambda s, t: fixed_time(s, t)); lr_, lw = evaluate(learned)
    return dict(fixed_reward=fr, fixed_avg_queue=fw, learned_reward=lr_, learned_avg_queue=lw)

def plot_rewards(rewards, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = 20; sm = np.convolve(rewards, np.ones(k)/k, mode="valid")
    plt.figure(figsize=(7, 3.5)); plt.plot(rewards, alpha=.3, label="episode reward"); plt.plot(range(k-1, len(rewards)), sm, label="moving avg")
    plt.xlabel("training episode"); plt.ylabel("total reward"); plt.title("Q-learning reward curve"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

if __name__ == "__main__":
    Q, rw = train(); print("episodes:", len(rw), "first20 avg:", np.mean(rw[:20]).round(1), "last20 avg:", np.mean(rw[-20:]).round(1))
    print(compare(Q)); print("Q-table states:", len(Q))
    for s in sorted(Q)[:6]: print(s, np.round(Q[s], 2))
