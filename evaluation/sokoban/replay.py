"""Replay a saved trajectory to GIF, or manually play a frozen test level."""
import argparse
import json
from pathlib import Path
from .env import Board, SokobanEnv, State
from .render import render
from .validate_dataset import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.cwd()
    parser.add_argument("--levels", type=Path, default=root / "data/eval_sokoban/levels.jsonl")
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", type=Path, help="GIF path for replay or PNG path for manual play")
    args = parser.parse_args()
    level = next(l for l in read_jsonl(args.levels) if l["level_id"] == args.level_id)
    board, initial = Board.from_dict(level["board"]), State.from_dict(level["initial_state"])
    if args.trajectory:
        if not args.output or args.output.suffix.lower() != ".gif":
            parser.error("Replay requires --output file.gif")
        env = SokobanEnv(board, initial, level["max_steps"])
        frames = [render(board, initial, **level["render"])]
        for row in read_jsonl(args.trajectory):
            if "action" not in row:
                continue
            if State.from_dict(row["state"]) != env.state:
                raise ValueError("Trace state does not match previous transition")
            env.step(row["action"])
            if State.from_dict(row["next_state"]) != env.state:
                raise ValueError("Trace next state does not match action")
            frames.append(render(board, env.state, **level["render"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=180, loop=0)
        print(str(args.output.resolve()))
    else:
        env = SokobanEnv(board, initial, level["max_steps"])
        keys = {"w": "up", "s": "down", "a": "left", "d": "right"}
        while not board.solved(env.state) and env.steps < env.max_steps:
            print("\n".join(board.ascii(env.state)))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                render(board, env.state, **level["render"]).save(args.output)
            key = input("w/a/s/d move, q quit: ").strip().lower()
            if key == "q":
                break
            if key in keys:
                _, event = env.step(keys[key])
                print(json.dumps(event))
        print("\n".join(board.ascii(env.state)))
        if args.output:
            render(board, env.state, **level["render"]).save(args.output)
        print("Solved" if board.solved(env.state) else "Stopped")


if __name__ == "__main__":
    main()
