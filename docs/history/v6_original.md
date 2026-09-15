# DualRSI Train V6 · AudioOnly4Attr

只保留 `原始音频 → MiDasheng caption → Qwen3-TTS 重建音频`。每轮更新顺序是 Captioner GRPO → TTS SFT。新版本从两个 base 开始，不加载 V5 adapter，不执行 caption-only、paired anchor、Captioner SFT 或 TTS GRPO。

## 已实现协议

- 扁平五字段：`transcript`、`gender`、`pitch_level`、`emotion`、`emotion_intensity`。枚举沿用现有方案。
- 中间 caption 只进行格式/类型/枚举规范化，不与原始标签或转录比较。合成使用它自己生成的 transcript，内部 language=`Auto`，不传 reference audio。
- 最终重建音频先用本地 Qwen3-ASR 检查内容；规范化 CER/WER/混合错误率 `<=0.10` 才进行属性打标。
- 性别和情绪强度：Qwen3.5-Omni-Plus；音高：Gemini-3.1-Pro-Preview；情绪：本地 emotion2vec-plus-large。评审模型与服务商配置冻结。
- 四属性等权精确匹配；原始 unknown 排除，重建 unknown 计 0；同组分母固定。
- `R=0.9*R_reconstruction+0.1*F`。内容不合格时重建分为 0，属性未请求，不是属性全错。
- `F=0`（不可解析），否则 `0.5+0.5*n_legal/5`。unknown 是合法枚举，格式分不判断属性事实。
- TTS SFT 在内容合格候选中按属性分选 top1，**无最低属性分门槛**。同分选 ASR 错误率低者，再按候选序号。
- SFT 输入为选中 caption，GT 为原始真实音频 codec。loss 沿用主码本 NLL + `0.3*`子码本 NLL。
- 每组奖励全相同则跳过 Captioner GRPO，但仍可产生 TTS SFT 记录。技术评审失败不能伪装成真实 0 分。

默认 411 条音频候选（源池 419 条，时长过滤 8 条）、10 轮、group=8、Captioner/TTS 推理每卡 batch=8、Captioner GRPO 重放 microbatch=1。参考不足的样本另列排除清单。

## 在 8 卡 H100 上运行

共享目录、模型、conda 环境需与当前 `/data/L202500147` 路径一致。直接在 tmux 前台执行，无需 nohup。调用现有部署的 Qwen/Gemini 密钥读取方式；不要把密钥写在命令或日志里。

先检查（不使用 GPU、不发 API 请求）：

```bash
cd /data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/run_v6.sh check "$PWD/runs/midasheng_v6_8gpu_run01"
```

批量比较：同组大小 8，比较 batch=4/8。使用 GPU，但没有 API 请求和训练更新。TTS 采用独立固定探测条件，不会用这些数据训练。

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/run_v6.sh batch-smoke "$PWD/runs/v6_batch_probe_8gpu_run01"
```

结果：`runs/v6_batch_probe_8gpu_run01/batch_comparison.json`。其中包含真实批量大小、串行回退、概率校验、吞吐时间与 worker 显存记录。批量比较可以用原命令重跑已完成阶段；不要在运行过程中修改框架源文件。

GPU 小规模训练验证：默认从长、中、短音频中选 16 条，group=8，训练一轮；会进行实际 API 打标、两种更新及 adapter 重新加载。新架构尚需在 GPU 服务器执行这一步。

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/run_v6.sh gpu-smoke "$PWD/runs/v6_smoke_8gpu_run01"
```

若 batch=8 的吞吐或一致性不适合，可在启动新 run 时加 `DUALISL_CAPTION_BATCH=4 DUALISL_TTS_BATCH=4`；两阶段可独立设置，group 仍为 8。smoke 若报告没有有效 GRPO/SFT 更新，不能视为更新验证通过：可在新 smoke 目录用 `DUALISL_SMOKE_RECORDS=32` 扩大覆盖。

正式训练（从 base 开始，不继承 smoke 权重）：

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
DUALISL_GROUP_SIZE=8 \
DUALISL_CAPTION_BATCH=8 \
DUALISL_TTS_BATCH=8 \
bash scripts/run_v6.sh train "$PWD/runs/midasheng_v6_8gpu_run01"
```

当前默认不人为限制显存。若某服务器每卡只有 32 GiB 可用，在**新 run 首次启动**命令前增加 `DUALISL_GPU_MEMORY_GIB=32`；该设置限制 PyTorch 分配器，CUDA/NCCL 等额外内存仍需余量。8 卡不会把单卡内存合并。并发使用其他训练时应按实际单卡预算配置。

中断恢复（配置与代码冻结，不能通过环境变量偷偷修改 batch/group/reward）：

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/run_v6.sh resume "$PWD/runs/midasheng_v6_8gpu_run01"
```

smoke 中断同样使用 `resume`，传 smoke 自己的目录。完成参考准备但还没开始训练时，也可使用 `prepare` → `train` 的同一路径。已开始训练的 run 使用 `resume`。

### 已运行任务：Qwen / Gemini 各 64 并发

新增根目录入口 `run_v6_api_workers.sh`，使用两个独立线程池。初版曾使用 Qwen 64、Gemini 64；实测 Qwen 触发 HTTP 429 后，当前默认已改为 Qwen 2、Gemini 64，并增加以下限速和持续重试。模型、prompt、解析、奖励及缓存身份沿用冻结实现。原入口仍使用冻结配置中的共享 4 worker。该扩展不改动冻结的 package/scripts 文件，因此已有阶段校验和缓存可以继续复用。

在**实际运行训练的服务器**的另一个 tmux 窗口执行停止命令：

```bash
cd /data/L202500147/Caption/DualRSI_Train_V6_AudioOnly4Attr
bash run_v6_api_workers.sh stop-labeling "$PWD/runs/midasheng_v6_8gpu_run01"
```

该命令只在 `attribute_labels` / `reference_labels` 阶段、没有本地子进程时，对精确匹配本 run 的主进程发送 SIGTERM，随后确认运行锁释放。旧线程池收到 Ctrl+C 时可能等待整个请求队列；停止命令避免这段等待。正在请求、尚未写入缓存的少量结果可能需要重试；已完成缓存保留。若阶段已经变为 GPU 更新，该命令会拒绝停止，此时在训练终端使用 Ctrl+C。

确认旧进程退出后，在 tmux 前台恢复：

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
DUALISL_QWEN_API_WORKERS=64 \
DUALISL_GEMINI_API_WORKERS=64 \
bash run_v6_api_workers.sh resume "$PWD/runs/midasheng_v6_8gpu_run01"
```

后续中断也使用这个入口恢复。无需改 `resolved_config.yaml`，其中 `api_workers: 4` 是原始冻结配置；实际并发以启动输出 `runtime_api_workers_per_model`、`label_cache/runtime_api_concurrency.json` 和事件日志为准。新任务也可使用此入口的 `train` 模式。扩展源码 SHA256 和每模型并发数会写入事件日志。128 并发的服务端承载能力需以实际响应验证；若限流频繁，可只通过两个运行时环境变量降低并发，再恢复。

### 无人值守：限速和持续 API 重试（当前推荐）

Qwen 现在默认 2 worker、每分钟最多发起 30 次请求，各线程和短重试共用限速器。收到 HTTP 429 后，该模型统一等待 60、120、240、最多 300 秒，并自动降低 Qwen 发送速率（不会因增加 worker 绕过限速）。该设置是根据当前限流现象采取的保守值，不代表服务商公布的额度。Gemini 保留 64 worker；如返回 429，同样按模型统一冷却。

原 evaluator 的三次短重试耗尽后，该任务继续按 60、120、240、480、最多每 900 秒重试，直到获得可用结果或用户主动停止。成功任务照常缓存，不重新计分；失败任务不会被伪造为 unknown 或 0 分，也不会直接导致本轮因 API EvaluationPending 退出。持续故障（包括密钥或余额问题）会持续等待，不保证服务未恢复时训练也能推进；GPU/数据/代码错误仍会正常报错。

旧进程退出后执行：

```bash
DUALISL_GPUS=0,1,2,3,4,5,6,7 \
DUALISL_QWEN_API_WORKERS=2 \
DUALISL_GEMINI_API_WORKERS=64 \
DUALISL_QWEN_API_RPM=30 \
bash run_v6_api_workers.sh resume "$PWD/runs/midasheng_v6_8gpu_run01"
```

启动输出 `persistent_api_retry: true` 表示新机制已加载。`api_rate_limit_wait` / `api_task_retry_wait` 是自动等待，不是程序退出；事件日志会记录 HTTP 状态码但不记录密钥、URL 或服务端原始错误正文。Ctrl+C 取消排队与退避等待，正在传输的少量请求仍受原超时约束。

查看状态或看板：

```bash
bash scripts/run_v6.sh status "$PWD/runs/midasheng_v6_8gpu_run01"
DUALISL_DASHBOARD_PORT=6007 \
bash scripts/run_v6.sh dashboard "$PWD/runs/midasheng_v6_8gpu_run01"
```

## 配置和阅读顺序

1. `configs/v6.yaml`：实际默认配置。
2. `dual_isl_train/orchestrator_v6.py`：数据准备、一轮闭环、两个更新与提交。
3. `dual_isl_train/schema_v6.py`：五字段、部分解析、TTS 条件。
4. `dual_isl_train/content_v6.py`、`reward_v6.py`：内容门槛、奖励和 SFT top1。
5. `dual_isl_train/labeling_v6.py`、`local_v6.py`：远程调用、本地 ASR/情绪、缓存和重试。
6. `scripts/midasheng_captioner_candidate.py`、`dual_isl_train/workers/qwen_voice_design.py`：复用并适配后的真实模型更新。

底层 worker 保留了部分历史辅助实现以维持概率重放/codec 接口；V6 调度入口和角色 action 白名单阻止启动旧训练阶段。`stage_runner.py` 只复用通用阶段执行，不继承 V4/V5 训练调度。

## 数据、断点与费用

`data/audio_manifest.jsonl` 与 `data/data_report.json` 是构建时 CPU 导出的清单。实际运行会重新核验音频与清单，在 run 中保存 `prepared/audio_manifest.jsonl`、`prepared/references.json`，并冻结数据身份。可通过 `data.exclude_audio_manifests` 提供独立评测音频清单进行哈希交集检查；没有配置清单时不会声称已自动完成所有 benchmark 的泄漏审计。

参考转录统一优先取 `training_data/labels_open_resolved_no_environment_check_trans_qwen3_asr_completed.jsonl`，兼容其中的旧字段名 `transcription`；缺失时再运行固定 Qwen3-ASR。411 条均已有参考转录，其中 52 条与旧角色清单文本不同，冲突 ID 已列入 `data/data_report.json`，不会混用不同来源。参考四属性由同一套固定专家生成并缓存。旧 paired 的人工 caption 不进入监督 loss。

内容规范化采用确定性规则：NFKC、英文小写、无关标点、标准英文基数词和有明确单位的中文数字转换；不使用 LLM 意译。模糊年份/连续数字读法不会强行合并。中文 CER/英文 WER/中英混合 token 编辑率分别记录，无短句额外放宽。

每个 API 任务最多 3 次尝试；字段级非法值降为 unknown。传输失败、整个 JSON 无效、返回模型不符等持久化为 pending。其他任务可先完成并缓存，重试耗尽后本轮停止在可恢复状态，不能提交未完成评分的轮次。服务恢复后用 resume。

原始参考打标最多 822 次 Qwen/Gemini 请求（411×2），是一次性准备成本。每轮请求量为内容合格重建音频数×2，最多 6,576 次，另加重试，缓存命中可减少。ASR/emotion2vec 本地运行。事件与 token usage 保存在 `label_cache/events.jsonl`，各轮统计保存 `round_NNN/summary.json`；真实人民币费用需要服务商结算单价，不能直接用请求数当金额。

主要输出：`logs/`、`status.json`、`run_state.json`、`round_NNN/collection/`、`round_NNN/training/`、`round_NNN/checkpoints/{caption_final,tts_final}`、`round_NNN/commit.json`、`latest.json`。只有两个 checkpoint 完整并校验后才提交本轮。共享权限脚本仅处理本 run 文件，不改旧 V5。

## 验证

```bash
/data/L202500147/miniconda3/envs/dualisl-critic/bin/python \
  -m unittest discover -s tests_v6 -v
```

CPU 测试验证奖励边界、10% 内容门槛、unknown 分母、无属性门槛 top1、平分 GRPO/SFT 区别、API 缓存与失败，以及 mock 两轮/中断恢复。mock 不证明 GPU 数值路径或训练效果，真实 GPU 检查结果以用户服务器的 smoke 为准。
