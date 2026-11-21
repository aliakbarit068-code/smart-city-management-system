"""Run every module, print results and save figures/CSVs into outputs/ (use these for report screenshots)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
from common import OUT, city_graph
import data, m1_agents, m2_search, m3_adversarial as m3, m4_informed as m4, m5_fuzzy as m5
import m6_ml, m7_ann, m8_rl, m9_cv_robotics as m9, m10_datascience as m10

def hdr(t): print("\n" + "="*8, t, "="*8)
df = data.generate()                                    # also writes data/traffic.csv

hdr("Module 1: intelligent agents")
for p, a, b, c, d in m1_agents.demo(): print(p, "| reflex:", a, "| model:", b, "| goal:", c, "s | utility:", d, "s")

hdr("Module 2: uninformed search (A -> G)")
for k, (p, c, e) in m2_search.demo().items(): print(f"{k}: {'->'.join(p)} cost={c} expanded={e}")
print("Vacuum world (start, plan):", m2_search.vacuum_world())

hdr("Module 3: adversarial search")
for b in (" "*9, "X O  O   "): print(m3.compare(b))
print("Traffic competition:", m3.traffic_game_tree())

hdr("Module 4: informed search & optimisation")
g = city_graph(); print("Greedy:", m4.greedy_best_first(g, "A", "G")); print("A*:", m4.astar(g, "A", "G")); print("Hill climbing:", m4.hill_climb_route(g, "A", "G"))
D = m4.dist_all(g); stops = ["A", "C", "E", "F", "G"]
print("TSP greedy:", m4.tsp_greedy(stops, D, "A")); print("TSP A*:", m4.tsp_astar(stops, D, "A"))
print("8 Queens (HC):", m4.eight_queens_hc()); print("8 Puzzle (A*) [moves, expanded]:", m4.puzzle_astar((8, 1, 3, 4, 0, 2, 7, 6, 5))[1:])
cmp_ = pd.DataFrame(m4.compare_algorithms()); print(cmp_.to_string(index=False)); cmp_.to_csv(f"{OUT}/search_comparison.csv", index=False)

hdr("Module 5: expert system & fuzzy logic")
print(m5.forward_chain({"accident", "high_density", "rain"}))
for d, w, e in [(.1, 10, 0), (.5, 40, 0), (.9, 90, 0), (.5, 40, 1)]: print(f"density={d} wait={w}s emergency={e} -> green {m5.fuzzy_green(d, w, e):.1f}s")
m5.plot_membership(f"{OUT}/fuzzy_membership.png")

hdr("Module 6: SVM & K-Means")
r = m6_ml.train_svm(df); print("SVM accuracy:", round(r["accuracy"], 3)); print(r["confusion"]); print(r["report"])
print(m6_ml.kmeans_patterns(df)[1]); m6_ml.plot_svm(f"{OUT}/svm.png")

hdr("Module 7: neural network")
a = m7_ann.train(df); print("ANN test accuracy:", round(a["accuracy"], 3)); m7_ann.plot_training(a, f"{OUT}/ann_training.png")

hdr("Module 8: reinforcement learning")
Q, rw = m8_rl.train(); print(m8_rl.compare(Q)); m8_rl.plot_rewards(rw, f"{OUT}/rl_rewards.png")
pd.DataFrame([(str(k), *v) for k, v in sorted(Q.items())], columns=["state(NS,EW,green,emerg)", "Q_keep", "Q_switch"]).round(2).to_csv(f"{OUT}/q_table.csv", index=False)

hdr("Module 9: computer vision & robotics")
img, truth = m9.make_traffic_image(); out, dets = m9.detect_vehicles(img)
from collections import Counter; print("truth:", dict(Counter(truth)), "detected:", dict(Counter(k for k, _ in dets)))
m9.save_detection_image(img, out, f"{OUT}/vehicle_detection.png")
res = m9.robot_run(m9.make_grid(), (0, 0), (14, 14)); print("robot reached:", res["reached"], "replans:", res["replans"]); m9.plot_robot(res, f"{OUT}/robot.png")

hdr("Module 10: data science")
clean = m10.preprocess(df); print(m10.summary(clean)); m10.make_charts(clean); print(m10.model_performance())
print("\nAll outputs saved in", OUT)
