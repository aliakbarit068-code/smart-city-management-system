"""Shared helpers: city graph used by several modules, output folder."""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

# City map (the example in the brief): A-B, A-C, B-D, C-E, D-E, E-F, D-F, F-G
# weights = travel cost in minutes
CITY_EDGES = [("A","B",4),("A","C",2),("B","D",5),("C","E",8),("D","E",2),
              ("D","F",7),("E","F",3),("F","G",1)]
# (x, y) coordinates for the heuristic; straight-line distance never exceeds the edge cost => admissible
CITY_POS = {"A":(2,6),"B":(1,4.5),"C":(3,4.5),"D":(1,3),"E":(3,3),"F":(2,1.5),"G":(2,0.5)}

def city_graph():
    g = {n: {} for n in CITY_POS}
    for a, b, w in CITY_EDGES:
        g[a][b] = w; g[b][a] = w
    return g
