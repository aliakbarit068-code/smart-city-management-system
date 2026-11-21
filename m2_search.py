"""Module 2: Uninformed search on the city map (BFS, DFS, UCS) + Vacuum World."""
from collections import deque
import heapq
from common import city_graph

def bfs(g, s, t):
    q, seen, expanded = deque([[s]]), {s}, 0
    while q:
        path = q.popleft(); n = path[-1]; expanded += 1
        if n == t: return path, cost(g, path), expanded
        for m in sorted(g[n]):
            if m not in seen: seen.add(m); q.append(path+[m])
    return None, float("inf"), expanded

def dfs(g, s, t):
    stack, seen, expanded = [[s]], set(), 0
    while stack:
        path = stack.pop(); n = path[-1]
        if n in seen: continue
        seen.add(n); expanded += 1
        if n == t: return path, cost(g, path), expanded
        for m in sorted(g[n], reverse=True):
            if m not in seen: stack.append(path+[m])
    return None, float("inf"), expanded

def ucs(g, s, t):
    pq, best, expanded = [(0, [s])], {}, 0
    while pq:
        c, path = heapq.heappop(pq); n = path[-1]
        if n in best and best[n] <= c: continue
        best[n] = c; expanded += 1
        if n == t: return path, c, expanded
        for m, w in g[n].items(): heapq.heappush(pq, (c+w, path+[m]))
    return None, float("inf"), expanded

def cost(g, path): return sum(g[a][b] for a, b in zip(path, path[1:]))

# ---------- Vacuum World (2 squares A/B) as a state-space search ----------
def vacuum_world():
    """State = (agent_location, dirtA, dirtB). Actions: Left, Right, Suck. Goal: no dirt. BFS."""
    start = ("A", True, True)
    def goal(s): return not s[1] and not s[2]
    def succ(s):
        loc, a, b = s
        yield "Left", ("A", a, b); yield "Right", ("B", a, b)
        yield "Suck", (loc, False if loc == "A" else a, False if loc == "B" else b)
    q, seen = deque([(start, [])]), {start}
    while q:
        s, plan = q.popleft()
        if goal(s): return start, plan
        for act, n in succ(s):
            if n not in seen: seen.add(n); q.append((n, plan+[act]))

def demo(s="A", t="G"):
    g = city_graph()
    return {name: f(g, s, t) for name, f in (("BFS", bfs), ("DFS", dfs), ("UCS", ucs))}

if __name__ == "__main__":
    for k, (p, c, e) in demo().items(): print(f"{k}: {'->'.join(p)} cost={c} expanded={e}")
    print("Vacuum world plan:", vacuum_world())
