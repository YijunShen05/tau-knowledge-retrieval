# HW3 baseline reproduction

Run commands from the project root. Keep the full official repository at `final/tau2-bench`. The launcher prepares a separate clone pinned to the public v1.0.0 release. Its commit is 17e07b1da2bbc0cadfddeea36412686e0604127b. The published Terminal trace records a different runner commit (26c0e46b7154f8dff2c2cdbadbcc5aa2ede91212), absent from the local Git object store. This pin approximates the paper-era implementation. Do not compare against later regraded leaderboard entries as though the scoring version were identical.

## Live BM25

The verified path uses Python 3.12 and uv (on PATH or in final/baseline/.tools/uv). Setup uses the frozen knowledge dependencies plus the pinned voice imports required by the text CLI. On Apple Silicon it can fetch the project-local PortAudio library if missing; other platforms need their system PortAudio development library for PyAudio. Both OpenRouter roles use only OPENROUTER_API_KEY or final/key.txt. The paper and Flash profiles use an OpenAI credential for the simulator. For Flash also configure the provider credential and set TAU_FLASH_MODEL to the exact LiteLLM model ID; use TAU_FLASH_API_BASE only if the agent endpoint needs an override. Keep credentials in the environment. Do not put them into a model ID or endpoint URL.

    python3 final/baseline/run.py setup
    python3 final/baseline/run.py smoke --profile flash
    python3 final/baseline/run.py dev --profile flash

Smoke uses one trial of task_098. Dev runs ten tasks once each; it includes the smoke task and is a separate batch. A successful technical smoke test yields an official results.json with messages, a termination reason and a scored reward (including zero). The launcher checks each expected task/trial pair. Run directories live under final/baseline/runs/. The paper profile uses the GPT-5.2 high agent with the same GPT-5.2 low simulator. Use --dry-run to review the exact official command.

    python3 final/baseline/run.py smoke --profile paper
    python3 final/baseline/run.py full --profile paper --dry-run

Dropping --dry-run from full starts 97 × 4 billed episodes. The reference BM25 row in Table 11 has pass^1/2/3/4 of 24.48%, 16.49%, 12.63%, 10.31%. A Flash-agent result belongs on its own row. The wrapper fixes seed 123 as an experimental choice (matching the archived Terminal trace header); the original BM25 run seed has not been verified. Set TAU2_DATA_DIR to the release data, as the wrapper does automatically.

## Recompute the published Terminal metrics offline

No provider credentials are required. The blob below is the unmodified 388-episode GPT-5.2 high banking trajectory file bundled in v1.0.0, at web/leaderboard/public/submissions/gpt-5-2_sierra_2026-02-26/trajectories/gpt-5.2_high_banking_knowledge_gpt-5.2_4trials.json. The scorer uses stored rewards and validates completeness. It does not rerun inference or the reward evaluator.

    git -C final/tau2-bench cat-file blob 875989833d28ab71423950ea77a96fb994c736b1 > /tmp/dm_paper_terminal_traces.json
    python3 final/baseline/score.py /tmp/dm_paper_terminal_traces.json --expected-tasks 97 --trials 4

99 of 388 episodes succeeded; 9 of 97 tasks succeeded in all four trials. The public Terminal breakdown in published-terminal-replay.json matches Table 11 to two decimals. For per-task successful trials c and total n, the pass^k estimator averages choose(c,k)/choose(n,k) over tasks.

For the authorized single-task OpenRouter configuration, run `python3 final/baseline/run.py smoke --profile openrouter` after setup. Both roles use OpenRouter openai/gpt-5.2. The wrapper loads OPENROUTER_API_KEY or final/key.txt at runtime. See task098-analysis.md for status and the archived comparison.

## Verified single-task result

The OpenRouter smoke run ended with user_stop and reward 0. Recheck offline from final/baseline (no inference charges):

    python3 score.py runs/openrouter-openrouter_openai_gpt_5_2-smoke/results.json --expected-tasks 1 --trials 1
    LITELLM_LOCAL_MODEL_COST_MAP=True tau2-v1.0.0/.venv/bin/python replay_task098.py runs/openrouter-openrouter_openai_gpt_5_2-smoke/results.json

The second command first scores the observed trace, then scores a separate in-memory diagnostic that changes the submitted account type and regenerates its tool receipt. Only the observed trace counts toward the baseline.

## DeepSeek V4.1 Flash 1017 · 四次配对实验

从项目根目录运行。每个题目和方法各写一份官方结果。十题检查点有意涵盖不同评分方式。审计、核账与离线重评分脚本使用官方环境中的 Python。只有 run.py 发起模型请求。

```sh
export TAU2_DATA_DIR="$PWD/final/baseline/tau2-v1.0.0/data"
export LITELLM_LOCAL_MODEL_COST_MAP=True
tau_python="$PWD/final/baseline/tau2-v1.0.0/.venv/bin/python"
python3 final/baseline/ds4x/run.py pilot --workers 6
"$tau_python" final/baseline/ds4x/audit.py
"$tau_python" final/baseline/ds4x/verify.py pilot
"$tau_python" final/baseline/ds4x/recheck.py
python3 final/baseline/ds4x/report.py pilot
python3 final/baseline/ds4x/run.py remaining --workers 10
"$tau_python" final/baseline/ds4x/verify.py full
python3 final/baseline/ds4x/report.py full
```

两阶段都可用同一条命令续跑。官方检查点按题号、trial 和 seed 跳过已评分的结果，包括 0 分。基础设施错误由官方续跑逻辑清理。费用取有效 simulation ID 下的 OpenRouter 回执，合并旧轨迹自带的 usage.cost，并按 generation ID 去重。原始文件在 `runs/ds1017-4x/task_XXX/{bm25,golden_retrieval}/`。每个目录的 results.json 保留全部四次的官方轨迹和评分，receipts.jsonl 保留模型用量。默认报告并列呈现双方 trial 0 和官方参考，右栏可展开作者用 Terminal 实跑的同题同种子轨迹。

当额度不足导致全量中断时，可用 verify.py snapshot 和 report.py snapshot 生成阶段快照。快照以两组均完成四次的题汇总 pass 指标，以所有已评分轨迹汇总费用。恢复可用额度后直接重新运行 remaining，官方检查点会跳过有效试次。

### DeepSeek 官方接口续跑

从项目根目录运行。客服由 DeepSeek 官方 `deepseek-flash` 提供，2026-11-15 对应 V4.1-Flash-1017。客服请求仅注册银行环境工具并使用 high 推理。GPT-5.2 low 模拟用户和适用的 GPT-4.1 裁判继续走 OpenRouter，因此两家接口均需可用。密钥分别从 `final/ds_key.txt` 和 `final/key.txt` 读取。

```sh
python3 final/baseline/ds4x/run.py native-one --workers 1
python3 final/baseline/ds4x/run.py native-remaining --workers 4
```

首条命令运行一个未完成的 trial。第二条补齐剩余试次。启动器同时查看两家渠道的有效检查点，按原题目、方法、trial 和 seed 跳过已评分轨迹。官方直连的结果写入 `runs/ds1017-native-4x/`，使用独立的 results.json 和 receipts.jsonl。核账和 HTML 报告会读取两个目录，检查同一 trial 与 seed 唯一，并在选中轨迹旁标出客服渠道。DeepSeek 直连回执缺少美元金额时，报告在可确定的非高峰时段按缓存命中、未命中与输出 token 估算，单独标出估算金额。
