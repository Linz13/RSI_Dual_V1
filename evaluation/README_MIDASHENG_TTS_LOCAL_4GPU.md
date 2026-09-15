# MiDasheng 第 6/9 轮：改用 GPU 4–7 的本地 TTS 评测

用户已停止八卡任务。新入口仅使用后四张卡，不计算第十二轮，不调用 API。

## 预检查和运行

使用此前能读取 checkpoint 的评测账号。确认旧 launcher 及其生成/评分 worker 已退出，再在目标服务器运行：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=4,5,6,7 bash run_midasheng_tts_local_4gpu.sh check
```

在 tmux 前台执行：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=4,5,6,7 \
bash run_midasheng_tts_local_4gpu.sh full \
  --output-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_4gpu_20260909_run01 \
  --source-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_20260909_run01
```

两轮并行：第 6 轮使用 GPU 4,5，第 9 轮使用 GPU 6,7；生成阶段各使用第一张卡（4、6），batch=8；本地评分每轮两套 Whisper/WVMOS，使用组内两张卡。没有使用物理 GPU 0–3，也没有把两台服务器的 GPU 合成一个任务。

## 已有音频复用与隔离

- 新目录与旧八卡目录分开，原八卡代码/身份/输出均保留。
- 首次 full 启动在旧 launcher 锁可用时，核对源 checkpoint 身份、generation identity、原源码 hash 及 WAV 完整性，将 manifest 中成功的音频**复制**到新目录并逐文件验证 SHA-256；不使用会连带修改旧文件的硬链接。
- 此次查看旧 manifest，第 6/9 轮分别记录 144/128 条成功 WAV；启动时以实际通过核验的文件为准。只复制记录为成功的音频，不把中断时未提交的文件当完整结果。
- `audio_import.json` 记录来源和复制哈希。之后沿用既有生成器补齐剩余音频，再计算 WER/WVMOS。沿用其断点随机采样语义，不声称恢复后的音频与不中断运行逐字节等价。
- 不复制旧本地评分，以免 GPU 调度或路径变化造成结果错配；这些已复用音频也会在新目录统一评分。
- 第一次导入后的同目录重跑复用新进度，不反复覆盖旧音频。数据/代码/checkpoint/GPU 分配变化会拒绝普通 resume。
- 旧目录的 advisory lock 只能检测持有锁的 launcher；用户须确认被单独遗留的 worker 也已退出。入口不会替用户 kill 其他任务。

## 输出与恢复

```text
/data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_4gpu_20260909_run01/full/
  rounds_06/{generation.log,local_metrics.log,local_summary.json,audio_import.json}
  rounds_09/{generation.log,local_metrics.log,local_summary.json,audio_import.json}
  summary.md
  summary.csv
  summary.json
```

中断后等原进程退出，原 full 命令重跑即可。手工重新汇总：

```bash
cd /data/L202500147/Caption/benchmark
EVAL_GPUS=4,5,6,7 \
bash run_midasheng_tts_local_4gpu.sh summarize \
  --output-root /data/L202500147/Caption/benchmark/midasheng_original_tts_local_r5_r8_4gpu_20260909_run01
```

两指标和评分口径与 [原八卡说明](README_MIDASHENG_TTS_LOCAL.md) 相同：每轮 1,645 条，WER 是逐样本百分数均值，WVMOS 是本地自动音质估计，Win Rate 保持空白。

如果 Qwen2.5 四卡训练也在同一台服务器，不要同时启动这两个入口。先完成本评测再启动训练；另一台服务器的独立 GPU 4–7 可同时使用。

## 当前验证

5 项 CPU/mock 检查通过，Bash 语法通过。旧八卡 checkpoint 的 commit 在本机仍不可读，完整资源 precheck 在此账号未通过，需目标账号执行；没有创建新评测输出目录或启动 GPU。原共享权重的四文件权限命令仍见原八卡说明，若同一目标账号本来能读则无需重复处理。
