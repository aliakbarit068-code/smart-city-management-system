"""Module 5: Expert system (forward chaining) + fuzzy traffic-signal controller (scikit-fuzzy)."""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------- Expert system: facts + rules + inference engine ----------------
RULES = [  # (name, conditions, conclusion)
    ("R1", {"emergency_present"}, "give_priority"),
    ("R2", {"rain", "high_density"}, "extend_green_and_warn_drivers"),
    ("R3", {"high_density", "long_wait"}, "longer_green"),
    ("R4", {"low_density"}, "short_green"),
    ("R5", {"accident"}, "reroute_traffic"),
    ("R6", {"reroute_traffic", "high_density"}, "alert_control_centre"),   # chains from R5
]
def forward_chain(facts):
    facts, fired, changed = set(facts), [], True
    while changed:
        changed = False
        for name, cond, concl in RULES:
            if cond <= facts and concl not in facts:
                facts.add(concl); fired.append((name, concl)); changed = True
    return facts, fired

# ---------------- Fuzzy controller ----------------
def build_controller():
    density = ctrl.Antecedent(np.arange(0, 1.01, .01), "density")
    waiting = ctrl.Antecedent(np.arange(0, 121, 1), "waiting")          # seconds
    emergency = ctrl.Antecedent(np.arange(0, 1.01, .01), "emergency")   # 0 none ... 1 present
    green = ctrl.Consequent(np.arange(0, 91, 1), "green")               # seconds
    density.automf(3, names=["low", "medium", "high"])
    waiting["short"] = fuzz.trimf(waiting.universe, [0, 0, 50])
    waiting["moderate"] = fuzz.trimf(waiting.universe, [20, 60, 100])
    waiting["long"] = fuzz.trimf(waiting.universe, [70, 120, 120])
    emergency["absent"] = fuzz.trimf(emergency.universe, [0, 0, .5])
    emergency["present"] = fuzz.trimf(emergency.universe, [.5, 1, 1])
    green["short"] = fuzz.trimf(green.universe, [0, 10, 25])
    green["normal"] = fuzz.trimf(green.universe, [20, 40, 60])
    green["long"] = fuzz.trimf(green.universe, [50, 70, 90])
    green["priority"] = fuzz.trimf(green.universe, [80, 90, 90])
    rules = [
        ctrl.Rule(density["low"] & emergency["absent"], green["short"]),
        ctrl.Rule(density["medium"] & emergency["absent"], green["normal"]),
        ctrl.Rule(density["high"] & emergency["absent"], green["long"]),
        ctrl.Rule(waiting["long"] & ~emergency["present"], green["long"]),
        ctrl.Rule(waiting["short"] & density["low"], green["short"]),
        ctrl.Rule(emergency["present"], green["priority"]),
    ]
    return ctrl.ControlSystem(rules), (density, waiting, emergency, green)

_CS, VARS = build_controller()
def fuzzy_green(density, waiting, emergency=0.0):
    sim = ctrl.ControlSystemSimulation(_CS)
    sim.input["density"], sim.input["waiting"], sim.input["emergency"] = density, waiting, emergency
    sim.compute(); return float(sim.output["green"])

def plot_membership(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 3))
    for a, v in zip(ax, VARS):
        for term in v.terms.values(): a.plot(v.universe, term.mf, label=term.label)
        a.set_title(v.label); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

if __name__ == "__main__":
    print(forward_chain({"accident", "high_density", "rain"}))
    for d, w, e in [(.1, 10, 0), (.5, 40, 0), (.9, 90, 0), (.5, 40, 1)]:
        print(d, w, e, "->", round(fuzzy_green(d, w, e), 1), "s")
