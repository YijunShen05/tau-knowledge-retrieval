# τ-Knowledge · Banking Retrieval

**When should a conversational agent search again or ask the customer?**

NYU Shanghai · Data Mining · Yijun Shen

This project studies retrieval decisions during banking support conversations. We reproduced the official BM25 and Gold protocols on all **97 tasks**, with **four independent trials per task and condition**. The support agent is **DeepSeek V4.1 Flash 1017** and the user simulator is **GPT-5.2 low**.

## Read the project

| Start here | Contents |
| --- | --- |
| [English proposal](final/proposal/proposal-en.pdf) · [中文提案](final/proposal/proposal-zh.pdf) | Research question, baseline, experiment setup and future work |
| [Task 006 dialogue example](examples/task006.md) | The author's reference, a successful DeepSeek run and a failed run |
| [Results for all 97 tasks](RESULTS.md) | BM25 and Gold scores, with links to each task's recorded trajectories |
| [Interactive Chinese report](final/ds4x-report.html) | Download the HTML and open it locally to compare conversations |
| [Experiment instructions](final/baseline/README.md) | Setup, model routing and runner commands |

## Completed baseline

| Retrieval | Successful trials | pass¹ | pass⁴ | Tasks passing all four trials |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 110 / 388 | **28.35%** | 7.22% | 7 / 97 |
| Gold | 193 / 388 | **49.74%** | 35.05% | 34 / 97 |

**pass¹** estimates single-trial success using all four trials. **pass⁴** is the fraction of tasks that succeed in every trial. Gold inserts the task's designated documents into the initial context. BM25 retrieves from the local banking knowledge base during the conversation.

The 21.39 percentage-point pass¹ gap motivates studying how the agent finds and applies policy evidence. Our proposed two-stage retrieval instruction and targeted reviewer are planned experiments.

## Read a recorded trajectory

Start with [the task index](RESULTS.md), choose a task and open its BM25 or Gold result file. Each file contains task definitions, experiment settings and a `simulations` array.

| JSON field | Meaning |
| --- | --- |
| `simulations[].trial` and `seed` | Independent trial number and random seed |
| `simulations[].messages` | Dialogue, tool arguments and tool responses |
| `simulations[].reward_info` | Official reward and scoring details |
| `simulations[].termination_reason` | How the episode ended |

The full experiment spans two agent-serving routes. Some tasks have trials in both directories, which the index links separately.

| Agent-serving route | Accepted trials | Recorded results |
| --- | ---: | --- |
| OpenRouter | 504 | [ds1017-4x](final/baseline/runs/ds1017-4x/) |
| DeepSeek native API | 272 | [ds1017-native-4x](final/baseline/runs/ds1017-native-4x/) |
| **Total** | **776** | **97 tasks × 4 trials × 2 conditions** |

Other run folders contain pilot and smoke-test experiments. The full-run scores use the two directories above. Banking profiles and account records are synthetic benchmark data.

## Reproduce the analysis

From the repository root, run the offline audit with Python.

```sh
python3 final/baseline/ds4x/proposal_audit.py
```

It checks the archived seeds, scores, token counts and tool-call IDs. For the pinned environment, official scoring and report generation, follow [the baseline setup instructions](final/baseline/README.md).

| Implementation | Entry point |
| --- | --- |
| Four-trial experiment launcher | [run.py](final/baseline/ds4x/run.py) |
| Native DeepSeek adapter | [native_deepseek.py](final/baseline/ds4x/native_deepseek.py) |
| Join the two serving routes | [shards.py](final/baseline/ds4x/shards.py) |
| Validate results and seeds | [verify.py](final/baseline/ds4x/verify.py) |
| Audit tokens and tool calls | [proposal_audit.py](final/baseline/ds4x/proposal_audit.py) |
| Generate the comparison report | [report.py](final/baseline/ds4x/report.py) |

## Experiment provenance

- Official environment version **v1.0.0**, commit `17e07b1da2bbc0cadfddeea36412686e0604127b`.
- Matched seeds **54907, 280679, 91421, 806309**, with limits of 200 steps and 10 errors.
- GPT-4.1 judges the task that includes a natural-language scoring criterion.
- The export omits opaque provider-encrypted reasoning payloads and system fingerprints. Provider request IDs are replaced with stable hashes, and local home paths with `/workspace`.
- Per-task statistics are in [full-verified.json](evidence/full-verified.json). Token and tool counts are in [usage-and-tools.txt](evidence/usage-and-tools.txt).

[Shi et al., τ-Knowledge, ICML 2026](https://proceedings.mlr.press/v306/shi26ah.html) · [Official implementation](https://github.com/sierra-research/tau2-bench) · [Upstream license](UPSTREAM-LICENSE)
