import pytest

from evaluation.game.dataset import make_game, validate_initial_frame, solution, stable_int, action_request
from evaluation.game.evaluate import run_episode
from evaluation.sokoban.evaluate import MemoizedPolicy


@pytest.mark.parametrize("environment", ["maze", "frozen_lake", "sokoban", "lane_racer", "parkour_runner"])
def test_published_style_reconstruction_and_complete_oracle_episode(tmp_path, environment):
    record = {"meta": {"environment":environment,"map_seed":42,"split":"eval","episode_step":0,
                       "state_index":0,"map_id":"fixture", "source_revision":"arcade-v2" if environment in {"lane_racer","parkour_runner"} else "game-v4"}}
    env = make_game(record)
    actions = solution(env)
    assert actions
    # Simulate missing initial images, as happens during published image deduplication.
    # 模拟发布去重后只保留中途状态的情况。
    if environment == "maze":
        for _ in range(min(2,len(actions)-1)):
            choices = env.optimal_actions()
            env.step(choices[stable_int(env.seed,env.steps,"trajectory") % len(choices)])
        record["meta"].update(state_index=env.steps,episode_step=env.steps)
    path = tmp_path/"published.png"
    env.observation().save(path)
    record["assets"]=[{"path":str(path)}]
    record["request"]={"state":{"messages":[{"role":"system","content":env.prompt()}]},
                       "questions":{"next_action":{"type":"choice","instructions":"Next action?","criteria":{a:a for a in env.actions}}}}
    reconstructed = make_game(record)
    validate_initial_frame(reconstructed,record)
    assert reconstructed.steps == 0
    class Policy:
        def __init__(self):self.actions=iter(actions)
        def choose(self,request):
            assert "targets" not in request and "meta" not in request
            action=next(self.actions)
            return {"action":action,"probabilities":{a:float(a==action) for a in env.actions}}
    result=run_episode(Policy(),record,tmp_path/"episode")
    assert result["success"] and result["steps_over_optimal"] == 1.
    assert result["optimal_action_matches"] == result["decisions"]


def test_game_memoization_accepts_plain_system_prompt_and_hashes_pixels(tmp_path):
    from copy import deepcopy
    from PIL import Image

    record = {"meta": {"environment": "maze", "map_seed": 42, "split": "eval"},
              "request": {"questions": {"next_action": {"type": "choice",
                          "instructions": "Next action?", "criteria": {"up": "up", "down": "down"}}}}}
    env = make_game(record)
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    env.observation().save(first)
    second.write_bytes(first.read_bytes())

    class Policy:
        calls = 0
        def choose(self, request):
            self.calls += 1
            return {"action": "up", "probabilities": {"up": 1., "down": 0.}}

    base = Policy()
    policy = MemoizedPolicy(base)
    request = action_request(record, env, first)
    original = deepcopy(request)
    assert not policy.choose(request)["cache_hit"]
    assert request == original
    # Same pixels under another filename reuse the result; system rules remain significant.
    # 相同像素即使文件名不同也复用结果，系统规则的变化仍会触发新推理。
    assert policy.choose(action_request(record, env, second))["cache_hit"] and policy.verified
    assert policy.choose(request)["model_calls"] == 0 and base.calls == 2
    changed_rules = deepcopy(request)
    changed_rules["state"]["messages"][0]["content"] += " Additional rule."
    assert not policy.choose(changed_rules)["cache_hit"] and base.calls == 3
    Image.new("RGB", (384, 384), "black").save(first)
    assert not policy.choose(request)["cache_hit"] and base.calls == 4
