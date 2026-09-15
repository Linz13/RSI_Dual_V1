# Qwen2.5-Omni-3B 两版 Captioner 评测

2026-09-07：已新增 Qwen2.5 thinker-only 推理后端和单节点 8 卡入口。本次只准备代码、CPU 验证及命令，未启动真实 GPU smoke/full。

## 范围与口径

入口：`run_qwen25_v1_v2_benchmarks_8gpu.sh`。独立 profile `qwen25_v1_v2` 只发现下列两个 run 的 r0–r9：

| 分支 | 相对 Caption 的 run 路径 | 本次盘点 |
|---|---|---|
| `qwen25_v1` | `DualISL_Train/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_20260903_run01` | r0–r4 有 commit 和 final 权重，但当前主机无权读取这五个 commit |
| `qwen25_rewardv2` | `DualISL_Train_RewardV2/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_reward_v2_20260903_run02` | r0–r8 已通过 commit/hash 核验；r9 有 caption_final 但无 commit，保持 pending |

每次 `run` 冻结当时的 checkpoint 清单，只评测 commit 路径及整个 checkpoint hash 匹配的 `caption_final`。没有 commit 的轮次为 pending；不可读、hash 错误为 invalid 并阻止启动，不会静默省略一个分支。后续轮次提交后重复 `run` 即可增量评测。当前预期为 14 个已提交模型、42 项 full，原版五个 commit 权限和哈希仍需在可读主机核验。

三套任务沿用现有 V2/V3 的数据、prompt、解析和评分：

- EmotionTalk：`standard_public`，1,929 条音频 × 四任务，共 7,716 条预测。
- ParaSpeechCaps：attr6 Scheme A，140 条样本、840 条属性记录。
- StyleCap/PromptSpeech MCQ：3,112 道题。

加载 `models/Qwen2.5-Omni-3B` 的 `Qwen2_5OmniThinkerForConditionalGeneration`，直接向 thinker 加载训练 LoRA；BF16、FlashAttention 2、贪心解码。EmotionTalk 和 StyleCap 默认 batch=1；8 张卡各运行一个独立 benchmark/checkpoint 任务，不是 torchrun/DDP。仅评测训练 checkpoint，不重复评测 base 或 TTS。

## 8 卡 H100 命令

前提：共享存储在目标节点同样挂载为 `/data/L202500147`，八张 GPU 可用。无需 activate，入口使用共享环境：

- `miniconda3/envs/qwen2_5-omni-3b-captioner`：Qwen2.5 推理。
- `miniconda3/envs/qwen3-captioner`：调度、辅助检查和部分评分。
- `miniconda3/envs/emotiontalk-metrics`：EmotionTalk 指标和 Java。
- 共享 preflight 还检查原有 `midasheng-captioner` 环境及 benchmark 缓存。

```bash
cd /data/L202500147/Caption/benchmark
export EVAL_GPUS=0,1,2,3,4,5,6,7
export LATER_CAPTION_EVAL_ROOT="$PWD/qwen25_v1_v2_caption_eval_runs_20260907_run01"

bash run_qwen25_v1_v2_benchmarks_8gpu.sh inventory
bash run_qwen25_v1_v2_benchmarks_8gpu.sh check
```

确认 `check` 成功后，在终端前台执行：

```bash
bash run_qwen25_v1_v2_benchmarks_8gpu.sh run
```

或者后台执行（与前台二选一）：

```bash
mkdir -p "$LATER_CAPTION_EVAL_ROOT"
nohup bash run_qwen25_v1_v2_benchmarks_8gpu.sh run \
  >"$LATER_CAPTION_EVAL_ROOT/launcher_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "PID=$! OUTPUT=$LATER_CAPTION_EVAL_ROOT"
```

`run` 先 smoke 再 full，已有身份一致且完整的任务会跳过，部分输出按原 runner 的 resume 规则恢复。完成后验证本次 frozen snapshot 的全部 full 任务并生成汇总；训练中尚未提交的轮次不计为评测完成。中断后保持同一个输出根，再执行 `run`。

```bash
bash run_qwen25_v1_v2_benchmarks_8gpu.sh status
bash run_qwen25_v1_v2_benchmarks_8gpu.sh summarize
bash run_qwen25_v1_v2_benchmarks_8gpu.sh paths
```

输出：`$LATER_CAPTION_EVAL_ROOT/summary_qwen25_v1_v2/results.{md,csv,json}`；分任务日志：`$LATER_CAPTION_EVAL_ROOT/logs/`。如需独立重评，另设全新输出目录。模型环境可用 `QWEN25_PY=/绝对路径/bin/python` 覆盖。

## 已知限制和权限处理

当前主机对原版 r0–r4 的 `commit.json` 返回 Permission denied，文件权限为 0600；V2 r0–r8 可读且 hash 正确。本次没有修改训练文件内容或权限。若目标服务器的 `inventory/check` 也报这个错误，需要在原训练服务器以实际文件所有者身份开放这些提交元数据的读取权限：

```bash
for n in 000 001 002 003 004; do
  chmod a+r "/data/L202500147/Caption/DualISL_Train/runs/dual_recursive_8gpu_h100_qwen2_5_omni_3b_20260903_run01/round_${n}/commit.json"
done
```

然后重新执行 `inventory/check`。如果共享存储拒绝 chmod，应由存储/训练账号管理方授予读取权限；不要通过跳过 commit/hash 核验来启动。权限变化不会改变文件内容哈希。

评分保留原实现；此前 MiDasheng EmotionTalk 的 CLAPScore 有文本张量长度不一致导致 unavailable 的记录。任务 complete 表示样本计数和模型身份核验通过，具体指标成功状态仍看 `scores.json` 与汇总。本次没有修改指标定义或修复该指标。

## 验证

已通过 Bash 语法、10 项调度器测试、2 项 Qwen2.5 thinker/adapter/批量生成模拟测试、9 项 StyleCap、15 项 ParaSpeechCaps 和 15 项 EmotionTalk 测试，共 51 项 CPU 测试。Qwen2.5 环境的 transformers/PEFT/FlashAttention/音频工具导入成功。真实 GPU 加载、smoke 及 full 留待目标服务器执行。
