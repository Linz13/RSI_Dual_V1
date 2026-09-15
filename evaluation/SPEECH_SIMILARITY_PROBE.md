# V5 第二轮音频相似度实验

固定使用 round_001 的 audio-only 数据：176 组、432 条语义有效候选。
每一对都是原音频与候选的 `reconstructed_audio_path`，不重新生成音频，不调用 API，不更新训练模型。

## 在 GPU 服务器运行

在 tmux 里前台执行：

```bash
cd /data/L202500147/Caption/benchmark
bash run_speech_similarity_probe.sh run \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpu-memory-gib 26
```

自动执行文件/权重校验 → 每卡最长音频所在组的 GPU smoke → 全量评分 → 汇总及盲听页。
每卡一个进程，先 WavLM 后 XLSR，两个模型不同时驻留显存。
26 GiB 是 PyTorch allocator 上限，不包含所有 CUDA 开销；启动要求每卡至少 28 GiB 空闲，为 32 GiB 可用预算留余量。
同组原音频特征只提取一次，四种评分共用两次编码结果；跨卡按音频时长分组分摊。

中断后重复原命令续跑；已完整落盘的组跳过。修改代码、输入、模型或卡数时，使用新的 `--output-dir`，避免混用旧结果。
只测试 GPU 而不跑全量可将 `run` 换成 `smoke`。
本机已完成 CPU 检查；GPU 显存峰值和速度须由 GPU smoke 实测。

## 比较内容

| 编码器（冻结） | SpeechBERTScore precision | 时间平均后 cosine |
|---|---|---|
| WavLM-Large | 主比较项 | 控制项 |
| wav2vec2 XLSR-53 | 控制项 | 主比较项 |

两模型均取 `hidden_states[14]`，FP32、eval、eager attention。
单声道平均、polyphase 重采样至 16 kHz，按 checkpoint feature extractor 归一化。
正式实验完整音频评分，不裁剪，不填充。
SpeechBERTScore：对生成音频的每帧，取与原音频所有帧的最大 cosine，然后对生成帧求平均；额外保存反向 recall。
cosine：同一份特征沿时间求平均，再比较两个向量。
实现已与安装的上游逐帧评分函数核对；波形归一化与上游 raw-wave 示例不同，因此这是受控比较配置，不冒充论文原配置复现。
原始分数范围为 [-1,1]，此实验不添加奖励权重或修改训练。

## 输出与判断

默认目录：`reports/v5_speech_similarity_round001_run01/`

- `report.md`：分数分布、组内跨度、第一名是否一致。
- `summary.json`、`candidates.json`：完整机器可读结果。
- `listen.html`：最多 20 组可直接试听的自包含页面，含原音频和打乱顺序的候选。
- `plan.json`：输入及模型哈希、版本、分片和预处理配置。
- `logs/`：每卡 smoke/full 日志；每 30 秒终端显示完成的编码器/组数。

同时汇总全部候选，以及缓存转录基本一致的候选。初筛规则为：中英文均做 NFKC、大小写、标点归一化后，字符编辑距离/原转录长度 ≤0.1。
当前 171/432 条通过；不是重新 ASR，也不代表人工核验。组内至少两条通过才参与该子集的排序比较。

盲听样本先抽 10 组固定随机组，再优先补 10 组主指标第一名不一致的组。先选择再展开分数。
下载试听选择 JSON 后可汇总：

```bash
bash run_speech_similarity_probe.sh analyze --votes /实际路径/listening_votes.json
```

结果分别记录随机组和分歧组的指标/人工选择一致情况。样本较少且混合抽样，一致率不应当作总体准确率。
组内分差大、与现有属性奖励一致，都不能单独证明指标好。重点看内容基本一致时，哪个指标更符合实际听感；本实验本身不能证明加入奖励会提升训练效果。

## CPU 验证命令

```bash
bash run_speech_similarity_probe.sh check
bash run_speech_similarity_probe.sh cpu-smoke
/data/L202500147/miniconda3/envs/qwen3-tts/bin/python -m unittest discover -s tests -p test_speech_similarity_probe.py -v
```

CPU smoke 只截取真实音频对前两秒以验证两个编码器及评分公式，不用于最终结果。
