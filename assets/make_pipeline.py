"""Generate the Cobalt pipeline diagram recipe.

A pulse travels frontier -> Cobalt -> verifier, then loops back as a rejection.
Emits assets/pipeline.json for the pixelbolt CLI.
"""

import json

W, H = 224, 62
BOX_W, BOX_H, BOX_Y = 60, 24, 6
BOXES = [
    ("FRONTIER", "decompose", 4),
    ("COBALT", "draft", 82),
    ("VERIFIER", "check", 160),
]

BG = "#0b0e16"
PANEL = "#121a28"
EDGE = "#2a3f5f"
INK = "#f2f0e4"
DIM = "#7d93aa"
ARROW = "#3f6fb5"
PULSE = "#ffdb85"
HOT = "#fff1ca"

# Where the pulse sits in each frame (x, y).
PATH = [
    (68, 18),
    (76, 18),
    (112, 18),
    (148, 18),
    (190, 18),
    (190, 40),
    (150, 52),
    (112, 42),
]


def base_ops():
    ops = [
        {"op": "rect", "x": 0, "y": 0, "w": W, "h": H, "color": BG},
    ]
    for title, caption, x in BOXES:
        ops += [
            {"op": "rect", "x": x, "y": BOX_Y, "w": BOX_W, "h": BOX_H, "color": PANEL},
            {"op": "rect", "x": x, "y": BOX_Y, "w": BOX_W, "h": BOX_H,
             "color": EDGE, "filled": False},
            {"op": "text", "x": x + BOX_W // 2, "y": BOX_Y + 4, "text": title,
             "color": INK, "align": "center"},
            {"op": "text", "x": x + BOX_W // 2, "y": BOX_Y + 14, "text": caption,
             "color": DIM, "align": "center"},
        ]
    # Arrows between the boxes.
    ops += [
        {"op": "line", "x": 64, "y": 18, "x2": 81, "y2": 18, "color": ARROW},
        {"op": "line", "x": 142, "y": 18, "x2": 159, "y2": 18, "color": ARROW},
        # Return path: down from VERIFIER, left, up into COBALT.
        {"op": "line", "x": 190, "y": 30, "x2": 190, "y2": 52, "color": ARROW},
        {"op": "line", "x": 190, "y": 52, "x2": 112, "y2": 52, "color": ARROW},
        {"op": "line", "x": 112, "y": 52, "x2": 112, "y2": 31, "color": ARROW},
        {"op": "text", "x": 151, "y": 54, "text": "reject", "color": DIM},
    ]
    return ops


frames = []
for i, (px, py) in enumerate(PATH):
    ops = base_ops()
    # Arrowhead pointing into the next box.
    ops += [
        {"op": "pixel", "x": px, "y": py, "color": HOT},
        {"op": "pixel", "x": px + 1, "y": py, "color": PULSE},
        {"op": "pixel", "x": px, "y": py - 1, "color": PULSE},
        {"op": "pixel", "x": px, "y": py + 1, "color": PULSE},
        {"op": "pixel", "x": px + 1, "y": py - 1, "color": PULSE},
        {"op": "pixel", "x": px + 1, "y": py + 1, "color": PULSE},
    ]
    frames.append({
        "name": f"p{i}",
        "duration": 120 if i % 2 == 0 else 100,
        "ops": ops,
    })

recipe = {
    "version": 1,
    "name": "cobalt-pipeline",
    "width": W,
    "height": H,
    "palette": {
        "bg": BG, "panel": PANEL, "edge": EDGE, "ink": INK,
        "dim": DIM, "arrow": ARROW, "pulse": PULSE, "hot": HOT,
    },
    "frames": frames,
}

with open("pipeline.json", "w") as f:
    json.dump(recipe, f, indent=1)
print(f"wrote pipeline.json  {W}x{H}  {len(frames)} frames")