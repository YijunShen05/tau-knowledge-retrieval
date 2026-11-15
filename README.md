# τ-Knowledge banking retrieval

NYU Shanghai · Data Mining HW3 · Yijun Shen

This project studies when a banking agent should narrow its search. The completed baseline compares local BM25 retrieval with the official Gold condition on all 97 banking tasks. DeepSeek V4.1 Flash 1017 is the agent and GPT-5.2 low is the user simulator. The proposed two-stage retrieval skill and reviewer are future experiments.

## Proposal and results

- [English proposal](final/proposal/proposal-en.pdf) · [中文提案](final/proposal/proposal-zh.pdf)
- [中文逐题轨迹报告](final/ds4x-report.html) — download and open the HTML locally to inspect all 97 tasks.
- [Task 006 examples](evidence/task006-traces.json) — the author’s Terminal reference and two complete DeepSeek BM25 trajectories.
- [Per-task scores](evidence/full-verified.json) · [token and tool-call counts](evidence/usage-and-tools.txt)

| Agent condition | Scored episodes | Successes | pass¹ | pass⁴ |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 388 | 110 | 28.35% | 7.22% |
| Gold | 388 | 193 | 49.74% | 35.05% |

## Raw trajectories

The two full-run directories contain the recorded runner trajectories. The export omits opaque provider-encrypted reasoning payloads and system fingerprints, replaces provider request IDs with stable hashes, and replaces local user home paths with /workspace. Each file preserves the messages, model responses where available, tool calls and results, reward breakdown, trial number and seed.

- [OpenRouter agent route](final/baseline/runs/ds1017-4x/) — 504 accepted episodes.
- [DeepSeek native agent route](final/baseline/runs/ds1017-native-4x/) — 272 accepted episodes.
- Open either directory, then a task, then `bm25` or `golden_retrieval`, and finally `results.json`. Read `simulations[].messages` for trajectories. The same task can span both routes. Join on task, retrieval condition and trial number; exclude infrastructure errors. Each of the resulting 776 slots is unique.
- Other directories under [runs](final/baseline/runs/) hold earlier tryouts, recovery runs and the single-task smoke test. They are excluded from the full-run denominator. The experiment uses synthetic banking profiles from the benchmark.

## Reproduce the analysis

Run these commands from the repository root. The first command reconciles the archived messages, receipts, token totals and tool-call IDs with the Python standard library. It does not contact a model provider.

```sh
python3 final/baseline/ds4x/proposal_audit.py
```

For the original environment and scorer, clone the upstream repository, then let the launcher prepare its pinned release and Python environment. See [the baseline instructions](final/baseline/README.md) for setup prerequisites.

```sh
git clone https://github.com/sierra-research/tau2-bench.git final/tau2-bench
python3 final/baseline/run.py setup
final/baseline/tau2-v1.0.0/.venv/bin/python final/baseline/ds4x/verify.py full
python3 final/baseline/ds4x/report.py full
```

The experiment uses upstream v1.0.0 at `17e07b1da2bbc0cadfddeea36412686e0604127b`. The archived author traces used Terminal retrieval. Tool calls in the DeepSeek BM25 and Gold comparison use the banking environment and local corpus. The simulator and evaluator use the permissions defined by each task. The selected author reference and other redistributions from the official repo are covered by [the upstream license](UPSTREAM-LICENSE).

[Shi et al., τ-Knowledge, ICML 2026](https://proceedings.mlr.press/v306/shi26ah.html) · [Official implementation and data](https://github.com/sierra-research/tau2-bench)
