"""Module 3: Minimax and Alpha-Beta on Tic-Tac-Toe + a traffic-competition game."""
import math

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def winner(b):
    for x, y, z in LINES:
        if b[x] != " " and b[x] == b[y] == b[z]: return b[x]
    return None

class Counter: n = 0

def minimax(b, player, c):
    c.n += 1
    w = winner(b)
    if w: return (1 if w == "X" else -1), None
    if " " not in b: return 0, None
    best, move = (-math.inf, None) if player == "X" else (math.inf, None)
    for i in [i for i, v in enumerate(b) if v == " "]:
        b[i] = player; v, _ = minimax(b, "O" if player == "X" else "X", c); b[i] = " "
        if (player == "X" and v > best) or (player == "O" and v < best): best, move = v, i
    return best, move

def alphabeta(b, player, a, be, c):
    c.n += 1
    w = winner(b)
    if w: return (1 if w == "X" else -1), None
    if " " not in b: return 0, None
    move = None
    if player == "X":
        v = -math.inf
        for i in [i for i, s in enumerate(b) if s == " "]:
            b[i] = "X"; s, _ = alphabeta(b, "O", a, be, c); b[i] = " "
            if s > v: v, move = s, i
            a = max(a, v)
            if a >= be: break
    else:
        v = math.inf
        for i in [i for i, s in enumerate(b) if s == " "]:
            b[i] = "O"; s, _ = alphabeta(b, "X", a, be, c); b[i] = " "
            if s < v: v, move = s, i
            be = min(be, v)
            if a >= be: break
    return v, move

def compare(board=None, player="X"):
    board = list(board if board is not None else " " * 9)
    c1, c2 = Counter(), Counter()
    v1, m1 = minimax(board[:], player, c1)
    v2, m2 = alphabeta(board[:], player, -math.inf, math.inf, c2)
    return dict(board=board, minimax=(v1, m1, c1.n), alphabeta=(v2, m2, c2.n))

# ---- Traffic competition: two junction agents choose green time; shared road capacity ----
def traffic_game_tree():
    """Two agents alternately choose green time from {10,20,30}. Shared road has 50 s of capacity;
    each agent's score = own flow - other flow penalty. Returns tree value with minimax & alpha-beta counts."""
    opts = (10, 20, 30)
    def payoff(a, b):
        used = min(a + b, 50); a_share = a/(a+b)
        return a_share*used - (1-a_share)*used*0.5 - 0.1*a      # MAX (agent A) perspective
    # explicit 2-ply tree: A (MAX) picks, then B (MIN) replies
    cA, cB = Counter(), Counter()
    best, move = -math.inf, None
    for a in opts:
        cA.n += 1; worst = min(payoff(a, b) for b in opts); cA.n += len(opts)
        if worst > best: best, move = worst, a
    # alpha-beta
    alpha, bmove = -math.inf, None
    for a in opts:
        cB.n += 1; beta = math.inf; pruned = False
        for b in opts:
            cB.n += 1; beta = min(beta, payoff(a, b))
            if beta <= alpha: pruned = True; break
        if beta > alpha: alpha, bmove = beta, a
    return dict(best_green_A=move, value=round(best, 2), minimax_nodes=cA.n, alphabeta_nodes=cB.n, ab_move=bmove)

if __name__ == "__main__":
    print(compare(" " * 9)); print(compare("X O  O   ".ljust(9)[:9]))
    print(traffic_game_tree())
