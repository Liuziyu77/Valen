"""Run complete games. Visual policies receive requests containing pixels and rules only."""
import argparse
from copy import deepcopy
import hashlib
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from . import VERSION
from .build_dataset import dump_json, dump_jsonl
from .dataset import action_request, sha256_file
from .env import ACTIONS, Board, SokobanEnv, State, state_hash
from .render import render
from .solver import static_dead_squares
from .validate_dataset import read_jsonl, require, replay


class RandomPolicy:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def choose(self, request):
        return {"action": self.rng.choice(ACTIONS), "probabilities": dict.fromkeys(ACTIONS, .25)}


class ReferencePolicy:
    """Privileged sanity check, explicitly excluded from model performance."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def choose(self, request):
        action = next(self.actions)
        return {"action": action, "probabilities": {a: float(a == action) for a in ACTIONS}}


class MemoizedPolicy:
    """Cache a stateless deterministic policy by visible pixels AND prompt.

    Never uses symbolic state, legal moves or the reference solution. The first
    cache hit is independently recomputed to check deterministic output.
    """
    def __init__(self, policy):
        self.policy, self.cache, self.verified = policy, {}, False

    def choose(self, request):
        visible = deepcopy(request)
        for message in visible["state"]["messages"]:
            for item in message["content"]:
                if item["type"] == "image_url":
                    item["image_url"]["url"] = "sha256:" + sha256_file(item["image_url"]["url"])
        # Preserve candidate insertion order, which affects model input order.
        key = hashlib.sha256(json.dumps(visible, ensure_ascii=False).encode()).hexdigest()
        calls = 0
        if key in self.cache:
            cached = self.cache[key]
            if not self.verified:
                repeated = self.policy.choose(request)
                if repeated["action"] != cached["action"]:
                    raise RuntimeError("Memoization verification failed: nondeterministic action")
                old, new = cached.get("probabilities"), repeated.get("probabilities")
                if (old is None) != (new is None) or (old is not None and (
                        set(old) != set(new) or any(abs(old[a] - new[a]) > 1e-6 for a in old))):
                    raise RuntimeError("Memoization verification failed: nondeterministic distribution")
                self.verified, calls = True, 1
            return dict(deepcopy(cached), cache_hit=True, model_calls=calls)
        result = self.policy.choose(request)
        self.cache[key] = deepcopy(result)
        return dict(result, cache_hit=False, model_calls=1)


class ValenPolicy:
    def __init__(self, checkpoint, device="cuda"):
        import torch
        from transformers import AutoProcessor
        from valen.data.compiler import Compiler
        from valen.modeling.model import build_model
        from valen.training.checkpoint import load_checkpoint
        from valen.evaluation.inference import predict
        checkpoint = Path(checkpoint).resolve()
        config = json.loads((checkpoint / "config.json").read_text())
        config.update(device=device, gradient_checkpointing=False)
        self.model = build_model(config)
        load_checkpoint(checkpoint, self.model)
        self.model.eval()
        self.compiler = Compiler(AutoProcessor.from_pretrained(config["model_path"], local_files_only=True),
                                 ".", config.get("max_length", 8192), config.get("media_kwargs"))
        self.media_kwargs = config.get("media_kwargs") or {}
        self.torch, self.predict = torch, predict

    def choose(self, request):
        # No labels, symbolic state, legal-action mask or solver context.
        compiled = self.compiler.compile({"group_id": "inference", "request": request})
        with self.torch.inference_mode():
            result = self.predict(self.model, compiled, temperature=1.0)["answers"]["next_action"]
        return {"action": result["choice"], "probabilities": result["probabilities"]}


class QwenPolicy:
    def __init__(self, model_path, device="cuda", media_kwargs=None):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, GenerationConfig
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="eager", output_loading_info=True)
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete pretrained model load: " + str(loading))
        self.model.to(device).eval()
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_new_tokens = 8
        self.generation_config.do_sample = False
        self.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.device = device
        self.media_kwargs = media_kwargs or {}

    def choose(self, request):
        content = request["state"]["messages"][0]["content"]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": content[0]["text"] + "\nChoose your next move. Reply with exactly one word: up, down, left, or right."},
            {"type": "image", "path": content[1]["image_url"]["url"]}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False, return_dict=True, return_tensors="pt",
                                                    processor_kwargs=deepcopy(self.media_kwargs))
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=self.generation_config)
        text = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        if text not in ACTIONS:
            raise ValueError("Expected exactly one direction, received: " + repr(text))
        return {"action": text, "probabilities": None}


def run_episode(policy, level, episode_dir, reference=None, save_frames=False, wall_timeout=1200):
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=False)
    board, initial = Board.from_dict(level["board"]), State.from_dict(level["initial_state"])
    env = SokobanEnv(board, initial, level["max_steps"])
    if board.solved(initial):
        raise ValueError("Evaluation initial state is already solved")
    seen, repeats, invalid = Counter({env.state: 1}), 0, 0
    dead_squares = static_dead_squares(board)
    deadlock = False
    reason, error, latencies = None, None, []
    model_calls, cache_hits, uncached_seconds = 0, 0, []
    started = time.monotonic()
    current_image = episode_dir / "current.png"
    trace_path = episode_dir / "trajectory.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        while env.steps < env.max_steps:
            if time.monotonic() - started >= wall_timeout:
                reason = "wall_timeout"
                break
            before, before_hash = env.state, state_hash(board, env.state)
            step_started = time.monotonic()
            image_path = episode_dir / ("{:04d}.png".format(env.steps) if save_frames else "current.png")
            render(board, before, **level["render"]).save(image_path)
            request = action_request(image_path.resolve(), level["language"])
            try:
                result = policy.choose(request)
                model_calls += result.get("model_calls", 1)
                cache_hits += int(result.get("cache_hit", False))
                action, probabilities = result["action"], result.get("probabilities")
                if action not in ACTIONS:
                    raise ValueError("Policy returned an unknown action")
                if probabilities is not None:
                    if set(probabilities) != set(ACTIONS) or any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
                        raise ValueError("Invalid policy probability distribution")
                if time.monotonic() - started >= wall_timeout:
                    reason = "wall_timeout"
                    trace.write(json.dumps({"step": env.steps, "request_state_id": before_hash,
                                             "termination": reason, "action_not_executed": action}) + "\n")
                    break
                _, event = env.step(action)
            except Exception as exc:
                reason, error = "policy_error", type(exc).__name__ + ": " + str(exc)
                trace.write(json.dumps({"step": env.steps, "request_state_id": before_hash, "error": error}) + "\n")
                break
            elapsed = time.monotonic() - step_started
            latencies.append(elapsed)
            if not result.get("cache_hit", False):
                uncached_seconds.append(elapsed)
            invalid += int(not event["moved"])
            repeats += int(env.state in seen)
            seen[env.state] += 1
            deadlock = deadlock or any(b in dead_squares for b in env.state.boxes)
            trace.write(json.dumps({"step": env.steps, "state": before.to_dict(), "state_id": before_hash,
                                    "action": action, "probabilities": probabilities, "event": event,
                                    "next_state": env.state.to_dict(), "next_state_id": state_hash(board, env.state),
                                    "cache_hit": result.get("cache_hit", False), "model_calls": result.get("model_calls", 1),
                                    "seconds": elapsed, "static_deadlock": deadlock}, ensure_ascii=False) + "\n")
            if event["solved"]:
                reason = "solved"
                break
    if reason is None:
        reason = "step_limit"
    render(board, env.state, **level["render"]).save(episode_dir / "final.png")
    # Retain final and initial frames by default; the trace can render every step.
    render(board, initial, **level["render"]).save(episode_dir / "initial.png")
    if current_image.exists():
        current_image.unlink()
    solved = reason == "solved"
    distance = reference["optimal_distance"] if reference else None
    result = {"level_id": level["level_id"], "group_id": level["group_id"], "solved": solved,
              "steps": env.steps, "pushes": env.pushes, "invalid_moves": invalid,
              "repeated_states": repeats, "static_deadlock_proven": deadlock,
              "termination": reason, "error": error, "seconds": time.monotonic() - started,
              "optimal_distance": distance, "difficulty": reference.get("difficulty") if reference else None,
              "boxes": len(initial.boxes), "language": level["language"], "theme": level["render"]["theme"],
              "steps_over_optimal": env.steps / distance if solved and distance else None,
              "mean_step_seconds": sum(latencies) / len(latencies) if latencies else None}
    result.update(model_calls=model_calls, cache_hits=cache_hits,
                  uncached_decisions=len(uncached_seconds), uncached_seconds=sum(uncached_seconds))
    dump_json(episode_dir / "result.json", result)
    return result


def wilson(successes, total):
    if not total:
        return [0.0, 1.0]
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(results):
    count, solved = len(results), sum(r["solved"] for r in results)
    successful = [r for r in results if r["solved"]]
    total_steps = sum(r["steps"] for r in results)
    ratios = [r["steps_over_optimal"] for r in successful if r["steps_over_optimal"] is not None]
    return {"games": count, "solved": solved, "success_rate": solved/count if count else None,
            "success_rate_wilson95": wilson(solved, count),
            "mean_success_steps": sum(r["steps"] for r in successful)/len(successful) if successful else None,
            "mean_success_steps_over_optimal": sum(ratios)/len(ratios) if ratios else None,
            "total_steps": total_steps, "invalid_action_rate": sum(r["invalid_moves"] for r in results)/total_steps if total_steps else None,
            "static_deadlock_rate_lower_bound": sum(r["static_deadlock_proven"] for r in results)/count if count else None,
            "games_with_repeats": sum(r["repeated_states"] > 0 for r in results),
            "termination_counts": dict(Counter(r["termination"] for r in results)),
            "mean_episode_seconds": sum(r["seconds"] for r in results)/count if count else None}


def full_report(config, results):
    report = {"config": config, "overall": summarize(results), "buckets": {}}
    report["overall"].update(model_calls=sum(r.get("model_calls", r["steps"]) for r in results),
                             cache_hits=sum(r.get("cache_hits", 0) for r in results))
    decisions = sum(r.get("uncached_decisions", 0) for r in results)
    report["overall"]["mean_uncached_decision_seconds"] = (sum(r.get("uncached_seconds", 0) for r in results)/decisions if decisions else None)
    for field in ("difficulty", "boxes", "language", "theme"):
        report["buckets"][field] = {str(value): summarize([r for r in results if r[field] == value])
                                      for value in sorted({r[field] for r in results}, key=str)}
    return report


def evaluate(eval_dir, output, policy_name="random", checkpoint=None, model_path=None,
             device="cuda", seed=0, save_frames=False, wall_timeout=1200, limit=None,
             num_shards=1, shard_index=0, memoize=False, media_config=None):
    eval_dir, output = Path(eval_dir).resolve(), Path(output).resolve()
    require(not (eval_dir / "BUILDING").exists(), "Test dataset is incomplete")
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        require(sha256_file(eval_dir / name) == digest, "Test input hash mismatch: " + name)
    levels = read_jsonl(eval_dir / "levels.jsonl")
    reference_rows = read_jsonl(eval_dir / "reference.jsonl")
    references = {r["level_id"]: r for r in reference_rows}
    require(len(levels) == len(references) == len(reference_rows) == manifest["games"], "Test game count mismatch")
    require({l["level_id"] for l in levels} == set(references), "Test/reference coverage mismatch")
    if limit is not None:
        require(0 < limit <= len(levels), "Invalid smoke-test limit")
        levels = levels[:limit]
    require(num_shards >= 1 and 0 <= shard_index < num_shards <= len(levels), "Invalid shard configuration")
    levels = levels[shard_index::num_shards]
    require(policy_name in ("random", "oracle", "valen", "qwen"), "Unknown policy")
    require(wall_timeout > 0, "Wall timeout must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite run: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    policy = None
    identity = {"policy": policy_name}
    if policy_name == "valen":
        require(checkpoint is not None, "--checkpoint is required")
        policy = ValenPolicy(checkpoint, device)
        identity.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(Path(checkpoint)/"checkpoint.pt"))
        identity["media_kwargs"] = policy.media_kwargs
    elif policy_name == "qwen":
        require(model_path is not None, "--model-path is required")
        media_kwargs = json.loads(Path(media_config).read_text()).get("media_kwargs", {}) if media_config else {}
        policy = QwenPolicy(model_path, device, media_kwargs)
        identity.update(model_path=str(Path(model_path).resolve()))
        identity.update(media_kwargs=media_kwargs, model_config_sha256=sha256_file(Path(model_path)/"config.json"),
                        model_weights_sha256={p.name: sha256_file(p) for p in sorted(Path(model_path).glob("*.safetensors"))})
    elif policy_name == "random":
        policy = RandomPolicy(seed)
    if memoize:
        require(policy_name in ("valen", "qwen"), "Memoization is only for stateless deterministic model policies")
        policy = MemoizedPolicy(policy)
    config = {"version": VERSION, **identity, "eval_dir": str(eval_dir), "seed": seed,
              "temperature": 1.0, "wall_timeout": wall_timeout, "save_frames": save_frames,
              "smoke_subset": limit is not None, "privileged_reference": policy_name == "oracle",
              "dataset_manifest_sha256": sha256_file(eval_dir/"manifest.json"),
              "num_shards": num_shards, "shard_index": shard_index, "memoize": memoize,
              "device": device, "runner_sha256": sha256_file(__file__),
              "timeout_semantics": "checked_between_policy_calls_and_before_action_execution"}
    dump_json(output / "config.json", config)
    results = []
    for i, level in enumerate(levels):
        if policy_name == "oracle":
            policy = ReferencePolicy(references[level["level_id"]]["solution_actions"])
        result = run_episode(policy, level, output / level["level_id"], references[level["level_id"]], save_frames, wall_timeout)
        results.append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(levels):
            print(json.dumps({"event": "game_evaluation", "policy": policy_name, "games": i+1,
                              "solved": sum(r["solved"] for r in results)}), flush=True)
    config["memoization_repeat_verified"] = policy.verified if memoize else None
    dump_json(output / "config.json", config)
    report = full_report(config, results)
    dump_jsonl(output / "episodes.jsonl", results)
    dump_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.cwd()
    parser.add_argument("--eval-dir", type=Path, default=root / "data/eval_sokoban")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("random", "oracle", "valen", "qwen"), default="random")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--wall-timeout", type=float, default=1200)
    parser.add_argument("--limit", type=int, help="Smoke subset only; omit for the full dataset")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--memoize", action="store_true")
    parser.add_argument("--media-config", type=Path, help="Qwen baseline: read media_kwargs from the same checkpoint config")
    args = parser.parse_args()
    report = evaluate(args.eval_dir, args.output, args.policy, args.checkpoint, args.model_path,
                      args.device, args.seed, args.save_frames, args.wall_timeout, args.limit,
                      args.num_shards, args.shard_index, args.memoize, args.media_config)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
