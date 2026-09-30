# JevBench public evaluation

This evaluates local Valen checkpoints on the public tasks from
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), pinned to
`bb05a335bc809e61b20c0f745d25499a82b326fc`.
The snapshot has 48 easy, 72 standard and 111 hard tasks: 231 decisions in total.
The private tasks and imported judge tier are unavailable in the public release.

Run from the Valen repository root with its Python environment activated:

```bash
# Default destination: data/jevbench.
python -m evaluation.jevbench.dataset

python -m evaluation.jevbench.evaluate \
  --checkpoint next-generation-qwen/mixer/joint/latest \
  --output JevBench-exp/runs/mixer

python -m evaluation.jevbench.compare \
  --runs JevBench-exp/runs/bilinear JevBench-exp/runs/mlp \
         JevBench-exp/runs/role_mlp JevBench-exp/runs/mixer \
  --output JevBench-exp/comparison
```

The downloader's `--output` and evaluator's `--dataset` select another dataset
directory. Set `VALEN_JEVBENCH_DATA` to change the default dataset directory for
all JevBench commands. The evaluator accepts `--device`, `--files easy original hard`,
`--max-length`, `--warmup` (unmeasured calls per question type), `--limit`
(explicitly marked as a smoke subset), and `--resume`. Each task is evaluated
as an independent request on one device. Run models sequentially on the same
device for comparable latency. Resume requires the same checkpoint, dataset,
source files, hardware and evaluation settings. Existing outputs are never
silently overwritten; a truncated final prediction can be recovered.

## Input mapping

- Keep the frozen task and Choice criteria insertion order.
- Serialize structured state as JSON with sorted keys and Unicode preserved.
- Map Valen Noul `true/false` to JevBench `yes/no`. Carry the benchmark's custom
  truth criteria in the question instructions because Qwen Noul uses fixed
  candidate descriptions.
- Keep Score descriptions as ordered levels. Each level remains a separate
  Qwen branch.
- Do not put expected labels, rationales, provenance, exact gold distributions
  or task IDs into the prompt. Do not fit a calibration temperature to this test
  set; the temperature is fixed at 1.

## Scoring and outputs

`_upstream/` vendors the upstream task validator, distribution scorer and metric
aggregator unchanged, with their MIT license and revision attribution. This
keeps the evaluator independent of a separate JevBench installation.
The pinned scorer uses **argmax for Score accuracy**, with lexical tie breaking;
expected-value MAE is separate. An older upstream metric docstring describes
rounded expected-value accuracy, but the executed `score_task` is authoritative.
Invalid distributions and inference failures remain incorrect for accuracy;
calibration uses available valid native probabilities. ECE uses the largest
probability, not Valen's distribution-concentration `confidence` field.

Each run writes:

```text
run_manifest.json     Source, checkpoint and data hashes; runtime settings
predictions.jsonl     One outcome per task, including failures and timing
metrics.json          Upstream metrics, tier/type breakdowns, diagnostics
complete.json         Full selected-task coverage and failure count
```

The report includes accuracy, macro accuracy, Brier, ECE, NLL, ordinal MAE,
paraphrase consistency and local latency. Ten hard tasks additionally provide
exact gold distributions; their TVD and distribution Brier are reported
separately from hard-label calibration. NLL floors probabilities at `1e-12`,
and the report records that convention. Chance-corrected tier diagnostics use
the option counts of the evaluated public tasks.

GPU work is synchronized. End-to-end latency includes compilation, forward and
native answer conversion; model loading, warmup and scoring are excluded.
There is no network hop or assumed production-load adjustment. No serving cost
or official leaderboard composite is inferred: the public subset and local
timing cannot reproduce the sealed leaderboard protocol.

```bash
python -m pytest -q tests/integration/test_jevbench.py
```

Store local experiment plans, source snapshots, data and results in
`JevBench-exp/`, which is excluded from Git.

## Laya and Intern-Decision baselines

`baselines.py` calls the authors' local inference implementations. Supply a
checkout of [Laya](https://github.com/NandhaKishorM/laya) with `--source` and the
English checkpoint in `models/laya`. For Intern-Decision, it loads the
`inference.py` shipped with the selected HF checkpoint (Python 3.12 required).
Record a checkout of [Intern-Decision](https://github.com/InternLM/Intern-Decision)
with `--source` as additional release provenance. Install native dependencies in
a separate environment under the ignored experiment directory.

```bash
python -m evaluation.jevbench.baselines \
  --adapter laya --model models/laya --source JevBench-exp/laya-source \
  --dataset JevBench-exp/data/JevBench --output JevBench-exp/baseline-runs/laya

python -m evaluation.jevbench.baselines \
  --adapter intern_hf --model models/Intern-Decision-2B \
  --source JevBench-exp/Intern-Decision --dataset JevBench-exp/data/JevBench \
  --output JevBench-exp/baseline-runs/Intern-Decision-2B

python -m evaluation.jevbench.compare_models \
  --runs JevBench-exp/baseline-runs/valen-bilinear \
         JevBench-exp/baseline-runs/laya JevBench-exp/baseline-runs/Intern-Decision-2B \
  --dataset JevBench-exp/data/JevBench --output JevBench-exp/baseline-comparison
```

Native adapters preserve the original structured state, typed question and
option order, omitting answers, IDs and provenance. Native prompt serialization
and descriptions can differ from Valen's compiler. Laya keeps its shipped
512-token question budget and temperature map; its runtime reports state
truncation in each prediction's `usage`. It also limits question/option text.
Intern-Decision rejects inputs above 8192 tokens and uses its released default
temperature. Its raw T=1 predictions and metrics are saved separately using the
same forward pass and the released calibration operation. No benchmark labels
are used for temperature selection.

Model weights are hashed against any recorded HF LFS hashes before execution;
the actual inference code and tokenizer/configuration hashes are also saved.
The cross-model comparer rescoring all predictions verifies coverage, input
evidence, native answer mapping and all aggregate metrics, then requires the
same frozen dataset, GPU model, evaluator source, warmup and runtime versions.
Models retain their own prompt templates, context budgets and calibration.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/integration/test_jevbench.py tests/integration/test_jevbench_baselines.py
```
