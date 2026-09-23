"""Deterministic vector-like tiles rendered into PNGs with Pillow."""
from PIL import Image, ImageDraw

THEMES = {
    "classic": {"wall": "#475569", "floor": "#f8fafc", "line": "#d5dce5", "goal": "#0f766e",
                "box": "#d99b45", "player": "#2563eb"},
    "warm": {"wall": "#625555", "floor": "#fff9ed", "line": "#e6ddcb", "goal": "#007f73",
             "box": "#d89655", "player": "#7345ba"},
}


def render(board, state, theme="classic", tile_size=44):
    board.validate(state)
    if theme not in THEMES or not 24 <= tile_size <= 128:
        raise ValueError("Unknown theme or tile size outside 24..128")
    colors = THEMES[theme]
    size = tile_size
    image = Image.new("RGB", (board.width * size, board.height * size), colors["wall"])
    draw = ImageDraw.Draw(image)
    for p in sorted(board.floors):
        r, c = divmod(p, board.width)
        x, y = c * size, r * size
        draw.rectangle((x, y, x + size - 1, y + size - 1), fill=colors["floor"], outline=colors["line"])
        if p in board.goals:
            pad = max(2, size // 14)
            draw.ellipse((x + pad, y + pad, x + size - pad - 1, y + size - pad - 1),
                         outline=colors["goal"], width=max(2, size // 14))
            draw.ellipse((x + size * .42, y + size * .42, x + size * .58, y + size * .58), fill=colors["goal"])
        if p in state.boxes:
            pad = size // 5
            rect = (x + pad, y + pad, x + size - pad, y + size - pad)
            draw.rectangle(rect, fill=colors["box"], outline="#603d24", width=2)
            draw.line((rect[0] + 3, rect[1] + 3, rect[2] - 3, rect[3] - 3), fill="#603d24", width=2)
            draw.line((rect[0] + 3, rect[3] - 3, rect[2] - 3, rect[1] + 3), fill="#603d24", width=2)
        if p == state.player:
            pad = size // 4
            draw.ellipse((x + pad, y + pad, x + size - pad, y + size - pad), fill=colors["player"], outline="#172033", width=2)
            for eye in (.40, .58):
                draw.ellipse((x + size * eye, y + size * .40, x + size * eye + 2, y + size * .40 + 2), fill="white")
    return image
