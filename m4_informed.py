"""Module 4: Informed search & optimisation: Greedy, A*, Hill Climbing, TSP, 8-Queens, 8-Puzzle."""
import heapq, itertools, math, random, time
from common import city_graph, CITY_POS

def h(n, t): (x1, y1), (x2, y2) = CITY_POS[n], CITY_POS[t]; return math.hypot(x1-x2, y1-y2)  # km <= minutes here

def greedy_best_first(g, s, t):
    pq, seen, exp = [(h(s, t), [s])], set(), 0
    while pq:
        _, path = heapq.heappop(pq); n = path[-1]
        if n in seen: continue
        seen.add(n); exp += 1
        if n == t: return path, sum(g[a][b] for a, b in zip(path, path[1:])), exp
        for m in g[n]:
            if m not in seen: heapq.heappush(pq, (h(m, t), path+[m]))

def astar(g, s, t):
    pq, best, exp = [(h(s, t), 0, [s])], {}, 0
    while pq:
        f, c, path = heapq.heappop(pq); n = path[-1]
        if n in best and best[n] <= c: continue
        best[n] = c; exp += 1
        if n == t: return path, c, exp
        for m, w in g[n].items(): heapq.heappush(pq, (c+w+h(m, t), c+w, path+[m]))

def hill_climb_route(g, s, t, restarts=0):
    """Hill climbing on a route: start from a random route, swap in neighbours that reduce cost."""
    from m2_search import bfs, cost
    path, _, _ = bfs(g, s, t); best = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-1):          # try to shortcut/replace each intermediate node
            for m in g[best[i-1]]:
                if m != best[i] and best[i+1] in g[m]:
                    cand = best[:i]+[m]+best[i+1:]
                    if cost(g, cand) < cost(g, best): best, improved = cand, True
    return best, cost(g, best)

# ---------------- TSP over a few city locations ----------------
def dist_all(g):
    from m2_search import ucs
    nodes = list(g); return {(a, b): ucs(g, a, b)[1] for a in nodes for b in nodes if a != b}

def tsp_greedy(stops, D, start):
    tour, cur, left = [start], start, set(stops)-{start}
    while left:
        nxt = min(left, key=lambda x: D[(cur, x)]); tour.append(nxt); left.remove(nxt); cur = nxt
    tour.append(start); return tour, sum(D[(a, b)] for a, b in zip(tour, tour[1:]))

def tsp_astar(stops, D, start):
    """A* over (current, visited) states; h = cost of the cheapest hop to an unvisited city (admissible)."""
    others = [s for s in stops if s != start]
    def hh(cur, vis):
        rem = [x for x in others if x not in vis]
        if not rem: return D[(cur, start)]
        return min(D[(cur, x)] for x in rem)       # cheapest next hop: never over-estimates
    pq, best = [(hh(start, frozenset()), 0, start, frozenset(), [start])], {}
    while pq:
        f, g_, cur, vis, path = heapq.heappop(pq)
        if len(vis) == len(others) and cur == start and len(path) > 1: return path, g_
        if len(vis) == len(others) and cur != start:
            heapq.heappush(pq, (g_+D[(cur, start)], g_+D[(cur, start)], start, vis, path+[start])); continue
        if (cur, vis) in best and best[(cur, vis)] <= g_: continue
        best[(cur, vis)] = g_
        for x in others:
            if x not in vis:
                nv = vis | {x}; ng = g_+D[(cur, x)]
                heapq.heappush(pq, (ng+hh(x, nv), ng, x, nv, path+[x]))

# ---------------- 8 Queens with Hill Climbing ----------------
def conflicts(state): return sum(1 for i in range(8) for j in range(i+1, 8)
                                 if state[i] == state[j] or abs(state[i]-state[j]) == j-i)
def eight_queens_hc(seed=0, max_restarts=100):
    rnd = random.Random(seed)
    for r in range(1, max_restarts+1):
        s = [rnd.randrange(8) for _ in range(8)]
        while True:
            cur = conflicts(s); best, move = cur, None
            for c in range(8):
                for v in range(8):
                    if v != s[c]:
                        t = s[:]; t[c] = v; k = conflicts(t)
                        if k < best: best, move = k, t
            if move is None: break
            s = move
        if conflicts(s) == 0: return s, r
    return s, max_restarts

# ---------------- 8 Puzzle with A* (Manhattan distance) ----------------
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
def manhattan(s): return sum(abs(i//3-(v-1)//3)+abs(i%3-(v-1)%3) for i, v in enumerate(s) if v)
def neighbours(s):
    z = s.index(0); r, c = divmod(z, 3)
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            t = list(s); t[z], t[nr*3+nc] = t[nr*3+nc], 0; yield tuple(t)
def puzzle_astar(start):
    pq, best, exp = [(manhattan(start), 0, start, [start])], {}, 0
    while pq:
        f, g_, s, path = heapq.heappop(pq)
        if s == GOAL: return path, g_, exp
        if s in best and best[s] <= g_: continue
        best[s] = g_; exp += 1
        for n in neighbours(s): heapq.heappush(pq, (g_+1+manhattan(n), g_+1, n, path+[n]))

def compare_algorithms(s="A", t="G"):
    from m2_search import bfs, dfs, ucs
    g = city_graph(); rows = []
    for name, f, props in (("BFS", bfs, ("Yes","fewest edges","O(b^d)","O(b^d)")),
                           ("DFS", dfs, ("Yes (finite)","No","O(b^m)","O(bm)")),
                           ("UCS", ucs, ("Yes","Yes (least cost)","O(b^(1+C*/e))","O(b^(1+C*/e))")),
                           ("Greedy", greedy_best_first, ("No (can loop)","No","O(b^m)","O(b^m)")),
                           ("A*", astar, ("Yes","Yes (admissible h)","O(b^d)","O(b^d)"))):
        t0 = time.perf_counter(); p, c, e = f(g, s, t); dt = (time.perf_counter()-t0)*1000
        rows.append(dict(algorithm=name, path="->".join(p), cost=c, nodes_expanded=e, ms=round(dt, 3),
                         complete=props[0], optimal=props[1], time_complexity=props[2], space_complexity=props[3]))
    return rows

if __name__ == "__main__":
    g = city_graph()
    print("Greedy", greedy_best_first(g, "A", "G")); print("A*", astar(g, "A", "G")); print("HC", hill_climb_route(g, "A", "G"))
    D = dist_all(g); stops = ["A", "C", "E", "F", "G"]
    print("TSP greedy", tsp_greedy(stops, D, "A")); print("TSP A*", tsp_astar(stops, D, "A"))
    print("8Q", eight_queens_hc())
    print("8P", puzzle_astar((1, 2, 3, 4, 0, 6, 7, 5, 8))[1:])
    for r in compare_algorithms(): print(r)
