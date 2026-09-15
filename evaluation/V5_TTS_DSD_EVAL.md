# V5 第一轮 TTS 的 DSD 评测

## 更新：使用 gemini_audio.py 的 API

原 API 令牌额度耗尽，生成的 2,000 条 WAV 已全部完成。后续补跑评审请使用新入口：

```bash
cd /data/L202500147/Caption/benchmark
bash run_v5_dsd_audio_api.sh judge
```

此入口运行时从 `/data/L202500147/api/gemini_audio.py` 读取密钥和 BASE_URL，采用相同的
`requests.post`、`x-goog-api-key`、inline 音频调用方式。模型固定为 `gemini-2.5-pro`，
保留原评审提示词、temperature=0、安全设置及默认 thinking 行为，不采用示例中的描述音频提示词。
不使用 GPU，不重新生成音频；默认 16 并发，重复执行跳过已经成功的评审。

新结果独立保存在运行目录的 `evaluation_gemini_2_5_pro_audio_api/`，最终文件为
`dsd_summary.json`。旧失败记录保留。不要再用下面旧入口的 `judge` 命令补跑此次 API 评审。
新服务商和旧服务商不同，报告会记录这点；模型名称和评分规则保持一致。

2026-09-10 已用一条实际生成音频验证新接口：1/1 成功，约 24.5 秒，非 dry-run。
HTTP 协议、固定模型、thinking token 统计和错误处理也通过本地模拟检查。

入口：`run_v5_tts_dsd.sh`，调度代码：`v5_tts_dsd_eval.py`。

固定评测 `DualISL_Train_RewardV5_LabelRobust/runs/midasheng_v5_label_robust_10rounds_8gpu_run01/round_000/checkpoints/tts_final`。
运行前核对 `commit.json` 的目录哈希，防止误用未完成轮次或其他 checkpoint。
只生成 DSD：英文 1,000 条、中文 1,000 条。默认四卡各 500 条，每卡 batch=8。

## 在四卡服务器运行

```bash
cd /data/L202500147/Caption/benchmark

# 32 条 GPU 验证：每卡一个 batch；评审使用离线模拟，不产生 API 费用。
DSD_GPUS=4,5,6,7 DSD_BATCH_SIZE=8 DSD_GPU_MEMORY_GIB=32 \
  bash run_v5_tts_dsd.sh smoke

# 正式评测：生成 2,000 条音频，然后调用 Gemini API 并汇总。
DSD_GPUS=4,5,6,7 DSD_BATCH_SIZE=8 DSD_GPU_MEMORY_GIB=32 \
  nohup bash run_v5_tts_dsd.sh run > v5_tts_dsd_round000.log 2>&1 &

tail -f v5_tts_dsd_round000.log
```

GPU 编号沿用该服务器先前使用的物理编号 `4,5,6,7`。如果该容器实际暴露的是
`0,1,2,3`，请相应修改 `DSD_GPUS`。脚本不会终止其他训练或评测任务。

默认每个 TTS 进程的 **PyTorch allocator** 预算为 32 GiB，启动要求每张选中卡至少
有 34 GiB 空闲。CUDA 上下文及其他库的分配不完全受这个预算约束；同卡其他任务
也可能增加显存使用，因此这不是绝不 OOM 的保证。单条生成 OOM 时会停止本次
DSD 的工作进程，并保留已完成音频。可提高预算后用原命令续跑。显存预算、API
并发和物理卡号可以改变；分片数量、batch、checkpoint、评测代码和采样设置必须一致。

## 续跑与输出

默认输出目录：

```text
/data/L202500147/Caption/benchmark/InstructTTSEval-public/qwen3_voice_design/runs/v5_midasheng_round000_dsd_seed42
```

中断后重复原命令即可。已生成音频保留，已成功评审记录不重复调用 API；已记录
的失败评审会重试。生成已全部完成时，`run` 会跳过 GPU 阶段。不要同时向同一输出目录
启动两个运行实例；调度器会加锁拒绝重复运行。`smoke` 位于独立的 `smoke/` 子目录。

```bash
bash run_v5_tts_dsd.sh status  # 生成/评审数量
bash run_v5_tts_dsd.sh check   # CPU、数据、checkpoint、API 配置检查；无网络请求
bash run_v5_tts_dsd.sh generate # 仅 GPU 生成
bash run_v5_tts_dsd.sh judge    # 仅付费评审和汇总；无 GPU 需求
bash run_v5_tts_dsd.sh score    # 仅重新汇总；无 API 请求
```

也可在 mode 后传入自定义输出目录。单独运行 `judge`/`score` 时，仍要保留生成时的
分片数量、batch 设置。默认设置下直接执行上述命令即可。

- 分卡生成日志：`logs/generate_00_gpu4.log` 等。
- 合并清单：`generation_manifest.json`。
- 逐条裁判结果：`evaluation_gemini_2_5_pro/judge_results.jsonl`。
- **最终报告：`evaluation_gemini_2_5_pro/dsd_summary.json`**。
- 报告包含中英文 DSD 分数、覆盖率、与 base 的百分点差、token 用量和估算费用。

## 评测口径

沿用现有 base 的原始 DSD 指令、解码温度、top-p、最大长度、SDPA、Gemini-2.5-Pro、
官方评审提示词、temperature=0 和 inline 音频传输。默认 API 并发为 32，环境变量
`DSD_JUDGE_WORKERS` 可调整。沿用已部署的 API 配置，无需把密钥写进命令。

base 的已有结果直接复用：EN 82.20%、ZH 81.10%、双语 81.65%。本次仅代表 DSD
子任务，并非完整 InstructTTSEval 三任务结果。Gemini 使用稳定版，不标成论文
preview 模型的精确复现。GPU 分片改变了组批，本次按固定 batch 的样本键确定随机种子，
保证常规中断续跑不会重组剩余 batch；因此无法与历史 base 的全局随机流逐条一致。
遇到批量失败会拆分重试，日志会记录 `[BATCH][SPLIT]`；实际吞吐取决于拆分比例。

费用统计已补入 Gemini 的 thinking tokens，以官方标准单价做预算，人民币使用
1 USD≈7 CNY 的假定换算。历史测算约 ¥250，建议预算 ¥300；不包括 GPU 费用和
未记录的失败重试，实际金额以 API 服务商账单为准。

## 本机验证记录

2026-09-10：`check` 通过，第一轮 TTS 目录哈希与提交记录一致；DSD 共 2,000 条，
四卡计划的 batch=8；API 客户端配置可加载。5 项生成测试、13 项原有数据/评审/评分
测试、8 项新调度测试均通过（共 26 项），其中包含离线模拟评审及重复执行不增加记录的检查。
Shell 语法检查通过。本机没有可用 GPU，未执行真实 GPU 生成，也未发送付费 API 请求。
