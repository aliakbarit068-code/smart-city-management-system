"""Module 9: classic CV vehicle detection + emergency-robot simulation (A* on an occupancy grid).
No internet/camera needed: a synthetic road image is generated."""
import heapq, random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

CLASSES = {"car": (60, 30), "bus": (120, 36), "truck": (95, 34), "motorcycle": (24, 14)}
COLORS = [(200, 60, 60), (60, 90, 200), (230, 200, 60), (60, 180, 90), (170, 170, 170), (150, 60, 170)]

def _rectangle(img, x0, y0, x1, y1, color, thickness=-1):
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x0 > x1 or y0 > y1:
        return
    if thickness < 0:
        img[y0:y1+1, x0:x1+1] = color
        return
    for offset in range(thickness):
        img[y0+offset:min(y0+offset+1, y1+1), x0:x1+1] = color
        img[max(y1-offset, y0):y1+1, x0:x1+1] = color
        img[y0:y1+1, x0+offset:min(x0+offset+1, x1+1)] = color
        img[y0:y1+1, max(x1-offset, x0):x1+1] = color

def _blur(img, radius):
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius))).copy()
    rgb = Image.fromarray(img[:, :, ::-1])
    return np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius)))[:, :, ::-1].copy()

def to_rgb(img):
    """Convert the module's BGR image arrays to RGB for display."""
    return img[:, :, ::-1].copy()

def save_detection_image(img, out, path):
    comparison = np.hstack((to_rgb(img), to_rgb(out)))
    Image.fromarray(comparison).save(path)

def make_traffic_image(n=12, seed=3, size=(360, 640)):
    r = random.Random(seed); H, W = size
    img = np.full((H, W, 3), 90, np.uint8)
    for y in (H//3, 2*H//3): 
        for x in range(0, W, 60): _rectangle(img, x, y, x+30, y+1, (112, 112, 112))
    truth, boxes = [], []
    tries = 0
    while len(truth) < n and tries < 500:
        tries += 1; kind = r.choice(list(CLASSES)); w, h = CLASSES[kind]
        x, y = r.randint(5, W-w-5), r.randint(5, H-h-5)
        if any(x < bx+bw+6 and bx < x+w+6 and y < by+bh+6 and by < y+h+6 for bx, by, bw, bh in boxes): continue
        boxes.append((x, y, w, h)); truth.append(kind)
        _rectangle(img, x, y, x+w, y+h, r.choice(COLORS))
        _rectangle(img, x+w//5, y+3, x+4*w//5, y+h-3, (40, 40, 40), 2)
    img = _blur(img, .5)
    return img, truth

def classify(w, h):
    area = w*h
    if area < 600: return "motorcycle"
    if w > 105: return "bus"
    if w > 75: return "truck"
    return "car"

def detect_vehicles(img):
    """Threshold road-colour differences, clean the mask, and classify connected components by size."""
    road = np.full_like(img, 90)
    diff = np.clip(np.abs(img.astype(np.int32) - road).sum(2), 0, 255).astype(np.uint8)
    diff = _blur(diff, 1)
    mask = diff > 100
    mask = ndimage.binary_closing(mask, structure=np.ones((9, 9), dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    out, dets = img.copy(), []
    canvas = Image.fromarray(to_rgb(out))
    draw = ImageDraw.Draw(canvas)
    for label, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        ys, xs = region
        x, y = xs.start, ys.start
        w, h = xs.stop - x, ys.stop - y
        if w*h < 250 or h < 10: continue
        k = classify(w, h); dets.append((k, (x, y, w, h)))
        draw.rectangle((x, y, x+w, y+h), outline=(0, 255, 255), width=2)
        draw.text((x, max(y-10, 0)), k, fill=(0, 255, 255))
    out = np.asarray(canvas)[:, :, ::-1].copy()
    return out, dets

def density_from_detections(dets, capacity=20): return min(len(dets)/capacity, 1.0)

# ---------------- Robotics: autonomous emergency robot on a grid ----------------
def make_grid(n=15, seed=5, density=.2):
    r = random.Random(seed); g = [[1 if r.random() < density else 0 for _ in range(n)] for _ in range(n)]
    g[0][0] = g[n-1][n-1] = 0; return g

def robot_plan(grid, start, goal):
    n = len(grid); h = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    pq, best = [(h(start), 0, start, [start])], {}
    while pq:
        f, c, p, path = heapq.heappop(pq)
        if p == goal: return path
        if p in best and best[p] <= c: continue
        best[p] = c
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            q = (p[0]+dr, p[1]+dc)
            if 0 <= q[0] < n and 0 <= q[1] < n and grid[q[0]][q[1]] == 0:
                heapq.heappush(pq, (c+1+h(q), c+1, q, path+[q]))
    return None

def robot_run(grid, start, goal, new_obstacle_at_step=6):
    """Receives destination -> plans -> drives; a sudden obstacle appears and the robot replans -> reaches goal."""
    path = robot_plan(grid, start, goal); pos, trail, replans = start, [start], 0
    if path is None: return None
    i = 0
    while pos != goal:
        i += 1
        if i == new_obstacle_at_step and len(path) > 4:
            blocked = path[min(len(path)-1, path.index(pos)+2)]
            if blocked not in (goal, start): grid[blocked[0]][blocked[1]] = 1       # sensor detects new obstacle
        nxt = path[path.index(pos)+1]
        if grid[nxt[0]][nxt[1]] == 1:
            path = robot_plan(grid, pos, goal); replans += 1
            if path is None: return dict(trail=trail, replans=replans, reached=False, grid=grid)
            nxt = path[1]
        pos = nxt; trail.append(pos)
    return dict(trail=trail, replans=replans, reached=True, grid=grid)

def plot_robot_figure(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(res["grid"]), cmap="Greys")
    trail = np.array(res["trail"])
    ax.plot(trail[:, 1], trail[:, 0], "r-o", ms=3)
    ax.scatter([trail[0, 1]], [trail[0, 0]], c="g", s=90, label="start")
    ax.scatter([trail[-1, 1]], [trail[-1, 0]], c="b", s=90, label="goal")
    ax.legend()
    ax.set_title(f"Emergency robot (replans={res['replans']})")
    fig.tight_layout()
    return fig

def plot_robot(res, path):
    fig = plot_robot_figure(res)
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)

if __name__ == "__main__":
    img, truth = make_traffic_image(); out, dets = detect_vehicles(img)
    from collections import Counter
    print("truth:", Counter(truth)); print("detected:", Counter(k for k, _ in dets))
    save_detection_image(img, out, "outputs/vehicle_detection.png")
    res = robot_run(make_grid(), (0, 0), (14, 14)); print("robot reached:", res["reached"], "steps:", len(res["trail"]), "replans:", res["replans"])
