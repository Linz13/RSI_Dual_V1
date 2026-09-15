# MiDasheng 原版第 6、9 轮：8 卡本地 TTS 补评

2026-09-09 后续变更：用户已停止八卡评测，前四卡让给其他任务。当前请使用 [后四卡入口](README_MIDASHENG_TTS_LOCAL_4GPU.md)，新目录核验复制已有成功 WAV 后继续本地评测。本页八卡命令保留作历史记录。

用户当前主表只保留 **Base、第 3、6、9 轮**，第 12 轮已移除。Base 和第三轮的 WER/WVMOS 已有完整 1,645 条结果，因此本入口只补缺失的第 6、9 轮，不重复早期评测。

- 第 6 轮：原版 continuation 的 `round_005/checkpoints/tts_final`。
- 第 9 轮：同一 run 的 `round_008/checkpoints/tts_final`。
- 来源 run：`/data/L202500147/Caption/DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01`。
- 默认输出：`/data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01`。
- 当前状态（2026-09-09）：用户已在另一台 8 卡服务器启动默认 full 目录；共享输出中两轮身份文件、generation manifest、日志和音频均已出现，处于生成阶段，尚无完整本地评分汇总。本会话只读核验远端产物，没有启动额外 GPU 任务。交付时本机的 commit 权限阻塞是历史状态；用户启动后的权限与远端进程情况不能仅凭本机推断。

## 两个本地指标与 API 指标

本入口调用已有 `generate.py` 生成候选音频，再调用已有 `staged_local_metrics.py` 同时计算两个本地指标：

- **WER**：本地 `openai/whisper-large-v3` 转写生成音频，沿用 benchmark 文本规范化与编辑距离计算；先计算每条 WER 百分数，再取 1,645 条的算术平均。不能用训练环境的 `large-v3-turbo` 替代。
- **WVMOS**：本地 Wav2Vec2MOS 自动音质分数，不是人工 MOS。
- **Win Rate**：另一个 Gemini API judge 阶段；本入口没有调用该阶段，也不调用混合本地/API 的 `run_evaluation.sh`。输出中 Win Rate 保持 null/空白。

复用既有 strong-prompting 协议、完整 VoiceDesign base + 原版 TTS LoRA、seed=42、temperature=1、top-p=0.9、max-new-tokens=8192、SDPA，以及正式生成 batch=8。没有修改训练代码、已完成 run、历史音频或现有评分 workers。

## 1. 如遇不可读：在创建 checkpoint 的训练账号执行一次

现场两个权重和两个 commit 为 `0600`。当前读取账号确实收到 Permission denied。若目标评测账号也不能读取，需要训练文件所有者执行下列命令；只增加这四个文件的读取权限，不递归修改训练目录。

```bash
TTS_SOURCE_RUN=/data/L202500147/Caption/DualISL_Train/runs/dual_recursive_8gpu_h100_midasheng_from_r2_20260831_run01
chmod a+r \
  "$TTS_SOURCE_RUN"/round_{005,008}/commit.json \
  "$TTS_SOURCE_RUN"/round_{005,008}/checkpoints/tts_final/adapter_model.safetensors
```

如文件系统还施加额外 ACL/项目身份限制，需用有读取权限的账号运行评测；不要伪装 UID 或绕过校验。脚本不会自行更改源文件权限。

## 2. 目标 8 卡服务器预检查

服务器需要通过相同绝对路径访问共享 Caption 目录、模型和 Conda 环境。仅使用单节点 8 卡，建议原有 H100 80GB 服务器。

```bash
cd /data/L202500147/Caption/benchmark
bash run_midasheng_tts_local_8gpu.sh check
```

`check` 不加载模型、不启动 GPU 作业、不创建输出 run。它校验两轮 commit 的轮号、TTS 路径、完整 checkpoint 目录 hash（按原版规则排除 training_metrics）、adapter/base 身份、数据数量、离线模型缓存，并在两个环境下做 CPU import 检查。任一项失败即退出。

不加载模型的路径预览：

```bash
bash run_midasheng_tts_local_8gpu.sh plan
```

## 3. 在 tmux 前台运行全量

先进入自行创建的 tmux 窗口，再执行：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=0,1,2,3,4,5,6,7 \
bash run_midasheng_tts_local_8gpu.sh full \
  --output-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01
```

默认映射：

| checkpoint | 可见 GPU 组 | 音频生成 | 本地 WER/WVMOS |
|---|---|---|---|
| 第 6 轮（r5） | 0,1,2,3 | GPU 0，batch 8 | 4 worker，每卡一套 Whisper + WVMOS |
| 第 9 轮（r8） | 4,5,6,7 | GPU 4，batch 8 | 4 worker，每卡一套 Whisper + WVMOS |

两轮同时进行，各自生成完成后马上进入本地评分。现有生成器是单卡批量推理，因此生成时主要使用两张 GPU；两轮均进入评分时可使用全部八卡。不是八卡 DDP，也未新增改变采样顺序的生成分片。两轮合计生成/本地评测 3,290 条，不调用付费 API。

若想先做最短 GPU 闭环，可选执行下列命令，再运行上面的 full；smoke 每轮只取固定 seed 的一条样本，共两条，输出到独立 `smoke/` 子目录，不能将它的分数写入主表。

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
bash run_midasheng_tts_local_8gpu.sh smoke \
  --output-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01
```

## 4. 日志、汇总与中断恢复

每轮目录：

```text
midasheng_original_tts_local_r5_r8_20260909_run01/full/
  rounds_06/
    local_run_identity.json
    generation.log
    generation_manifest.json
    audios/
    local_metrics.log
    staged_evaluation/local_metadata.json
    staged_evaluation/local_metrics.jsonl
    local_summary.json
  rounds_09/                         # 相同结构
  summary.json
  summary.csv
  summary.md
```

终端输出指出当前阶段、日志路径和完成分数。另开 tmux 窗口查看：

```bash
tail -F /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01/full/rounds_{06,09}/{generation,local_metrics}.log
```

如果中断，确认原入口及子进程退出后，使用完全相同的 full 命令恢复。根目录锁拒绝重复入口；Ctrl-C/SIGTERM 会终止该入口启动的两个子进程组。数据、checkpoint、脚本源码、协议、GPU 分配或 Python 环境路径改变时拒绝复用原身份，需新输出根目录。

复用现有生成器的断点机制；未完成生成恢复时不保证随机采样的逐音频字节与不中断运行完全相同。完整音频存在且身份匹配时直接跳过生成；本地评分按成功记录跳过并补齐失败/缺失。已有本地评分不能与手工替换后的音频混用。

运行成功自动生成 `full/summary.md`、CSV、JSON。仅当每轮全部预期 ID 成功、WAV 可读、分数有限、生成与评分 metadata 匹配时写 `local_complete`。这是本地评测完成，不是包含 Gemini 的官方完整评测完成。人工填表时使用完成轮数 6、9 对应的 `wer_percent` 与 `wvmos`，Win Rate 继续留空。

原进程退出后也可不加载模型重新汇总：

```bash
cd /data/L202500147/Caption/benchmark
bash run_midasheng_tts_local_8gpu.sh summarize \
  --output-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01
```

## 实现和验证

- 入口：`run_midasheng_tts_local_8gpu.sh`。
- 预检查、两轮调度和纯本地汇总：`midasheng_tts_local_eval.py`。
- 两个实际评分均复用 `EmergentTTS-Eval-public/qwen3_voice_design/staged_local_metrics.py`。
- 针对性 CPU/mock 检查覆盖轮数映射、commit/hash、训练指标目录排除、均值汇总、失败/缺失/NaN/错音频拒绝、协议身份和仅本地命令路径。
- 交付检查：8 项针对性测试通过，Bash 语法通过；TTS 和本地评分环境的实际 CPU import 均成功，没有加载模型。当前主机 import 提示缺少 SoX/ffmpeg 及可选 flash-attn，既有入口使用 SDPA 和 WAV 路径；未据此安装或改动共享环境，目标节点实际音频闭环由可选 smoke/正式运行确认。
- 本地权限未解除前不能声称真实目标权重 hash 验证或 GPU 验收通过。没有运行正式评测。
