"""Module 1: Intelligent agents (simple reflex, model-based, goal-based, utility-based)."""

class SimpleReflexAgent:
    """Condition-action rules only. Table from the brief."""
    def act(self, percept):
        if percept["emergency"]: return "Give priority"
        d = percept["density"]
        return "Short green signal" if d < 0.35 else ("Normal green signal" if d < 0.7 else "Longer green signal")

class ModelBasedReflexAgent:
    """Keeps internal state (recent history) so one noisy reading does not flip the decision."""
    def __init__(self, window=3): self.history, self.window = [], window
    def act(self, percept):
        self.history.append(percept["density"]); self.history = self.history[-self.window:]
        smoothed = sum(self.history)/len(self.history)
        return SimpleReflexAgent().act({**percept, "density": smoothed})

class GoalBasedAgent:
    """Goal: clear queue below target. Simulates each green duration and picks the one that meets the goal."""
    def __init__(self, goal_queue=5): self.goal = goal_queue
    def act(self, percept):
        if percept["emergency"]: return 60
        queue = percept["density"]*60
        for green in (10, 20, 30, 45, 60):          # try options from shortest
            if queue - green*0.9 <= self.goal: return green
        return 60

class UtilityBasedAgent:
    """Maximises utility = -waiting_cost - energy_cost; picks best green time (seconds)."""
    def act(self, percept):
        queue = percept["density"]*60
        best, best_u = None, -1e9
        for green in (10, 20, 30, 45, 60):
            remaining = max(queue - green*0.9, 0)
            u = -2.0*remaining - 0.05*green*green/10 + (100 if percept["emergency"] and green == 60 else 0)
            if u > best_u: best, best_u = green, u
        return best

def demo():
    percepts = [{"density": .1, "emergency": False}, {"density": .5, "emergency": False},
                {"density": .9, "emergency": False}, {"density": .5, "emergency": True}]
    ra, mb, gb, ub = SimpleReflexAgent(), ModelBasedReflexAgent(), GoalBasedAgent(), UtilityBasedAgent()
    rows = []
    for p in percepts:
        rows.append((p, ra.act(p), mb.act(p), gb.act(p), ub.act(p)))
    return rows

if __name__ == "__main__":
    for p, a, b, c, d in demo():
        print(f"{p} | reflex={a} | model={b} | goal={c}s | utility={d}s")
