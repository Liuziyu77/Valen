"""Deterministic synthetic fixtures, not a model-quality benchmark."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import av
import numpy as np


def questions():
    return {
        "category": {"type": "choice", "instructions": "按材料选择交通灯颜色；无法判断时选择 unknown。",
                     "criteria": {"red": "红灯 / red light", "green": "绿灯 / green light", "unknown": "证据不足 / insufficient evidence"}},
        "go": {"type": "noul", "instructions": "规则：只有绿灯可以通行。当前可以通行吗？"},
        "risk": {"type": "score", "instructions": "Evaluate crossing risk: green is low, missing evidence is medium, red is high.",
                 "criteria": ["Low risk: green light", "Medium risk: light is not visible", "High risk: red light"]}}


def targets(color):
    return {"category": {"probabilities": {"red": float(color == "red"), "green": float(color == "green"), "unknown": float(color == "unknown")}},
            "go": {"probabilities": {"true": float(color == "green"), "false": float(color != "green")}},
            "risk": {"probabilities": {"0": float(color == "green"), "1": float(color == "unknown"), "2": float(color == "red")}}}


def frame(color):
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 8, 88, 120), fill="black")
    draw.ellipse((46, 16, 82, 52), fill="red" if color == "red" else "#222222")
    draw.ellipse((46, 76, 82, 112), fill="#00ff00" if color == "green" else "#222222")
    return image


def main():
    root = Path("data/smoke")
    root.mkdir(parents=True, exist_ok=True)
    frame("green").save(root / "green.png")
    with av.open(str(root / "red_to_green.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for i in range(16):
            video_frame = av.VideoFrame.from_ndarray(np.asarray(frame("red" if i < 8 else "green")), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    records = []
    for i, color in enumerate(("red", "green")):
        records.append({"group_id": f"synthetic-text-{i}", "request": {"state": "当前交通灯是红灯。" if color == "red" else "The traffic light is green.", "questions": questions()}, "targets": targets(color), "assets": []})
    for kind, filename in (("image", "green.png"), ("video", "red_to_green.mp4")):
        qs = questions()
        if kind == "video":
            for q in qs.values():
                q["instructions"] = "仅判断视频最后的状态。 " + q["instructions"]
        records.append({"group_id": "synthetic-" + kind, "request": {"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Observe the traffic light."}, {"type": kind + "_url", kind + "_url": {"url": filename}}]}]}, "questions": qs}, "targets": targets("green"), "assets": [{"path": filename, "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest()}]})
    records.append({"group_id": "synthetic-soft", "request": {"state": "有两位标注者认为低风险，一位认为中等风险。", "questions": {"soft": questions()["risk"], "missing": questions()["go"]}}, "targets": {"soft": {"probabilities": {"0": 2/3, "1": 1/3, "2": 0}, "label_source": "annotator_distribution", "votes": 3}, "missing": {"probabilities": None}}, "assets": []})
    records.append({"group_id": "synthetic-unlabeled", "request": {"state": "无可见交通灯。", "questions": {"missing": questions()["go"]}}, "targets": {"missing": {"probabilities": None}}, "assets": []})
    for filename, subset in (("train.jsonl", records), ("text.jsonl", [r for r in records if isinstance(r["request"]["state"], str)])):
        (root / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    print(f"Generated {len(records)} states: text/image/video; Choice/Noul/Score; soft/missing targets")


if __name__ == "__main__":
    main()
