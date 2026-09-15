# ParaSpeechCaps 测试 benchmark

这里保存的是论文 *Scaling Rich Style-Prompted Text-to-Speech Datasets*
用于主要 TTS 实验的 `test` split，而不是完整的 2,709 小时训练集。

## 先看这里

- `preview.html`：最直观的可搜索表格，包含 style prompt、transcript、标签、来源路径；其中 26 条 EARS 音频可直接播放。
- `data/test.jsonl`：246 条标注，适合脚本逐行读取。
- `data/test.csv`：完整 246 条标注；其中部分源音频尚不可用。
- `data/test_available.csv`：本地音频已验证可读的 142 条子集，captioning 评测请优先使用此文件。
- `audio/test/`：按 `benchmark_index` 命名的音频（例如 `014.wav`）。
- `audio/test/manifest.json`：246 条逐项下载状态、镜像来源和预处理说明。
- `data/test-00000-of-00001.parquet`：Hugging Face 官方原始文件，未修改。
- `summary.json`：样本、来源及目标标签统计。
- `source/official_repo`：论文官方 GitHub 仓库的浅克隆。

## Benchmark 是什么

论文从 PSC-Base 的 holdout 中按 rich tag 平衡抽取样本：每个 tag 最多 5 条，尽量覆盖更多说话人。
每条样本用一个 rich `tag_of_interest`、gender、可选的 pitch/speed，以及 clean noise tag 生成 style prompt。

当前官方 `test` split 的实际情况：

| 项目 | 数量 |
| --- | ---: |
| 样本 | 246 |
| 目标 rich tags | 50 |
| 参考音频总时长 | 1,587.696 秒（26.46 分钟） |
| VoxCeleb 来源 | 137 |
| Expresso 来源 | 83 |
| EARS 来源 | 26 |
| 本地可直接播放 | 142（Expresso 83、EARS 26、VoxCeleb 33） |

大部分 tag 各有 5 条；`pained` 有 4 条，`bored` 有 2 条。

## 音频说明

ParaSpeechCaps 官方只发布 caption、标签、transcript 和 `relative_audio_path`，不重新分发源音频。
参考音频来自 VoxCeleb、Expresso 和 EARS：

- EARS：工作区已经存在 `benchmark/ears_dataset`；26/26 条测试音频已通过符号链接匹配。
- Expresso：公开镜像中精确匹配到 83/83 条 VAD 分段音频。
- VoxCeleb：公开镜像中精确匹配到 33/137 条，继续补齐需要扫描或下载数十 GB 的源数据。

当前 142 条均已逐文件验证可读，总时长约 955.159 秒（15.92 分钟）。
`data/test_available.csv` 的 `local_audio_path` 已填为绝对路径，可以直接交给 captioning 推理脚本。
评测时应只遍历该 CSV，并使用其中的 `benchmark_index` 与标注对应；不要将缺失的 104 条计为失败样本。

当前公共镜像音频与 caption、transcript 的样本身份是精确匹配的，但不保证完全复现论文的音频预处理：
VoxCeleb 尚未执行论文要求的响度归一化和 VoiceFixer，Expresso/EARS 的响度归一化状态也未完全核实。
因此该子集适合 captioning 流程验证和模型间同集比较，不应将结果表述为完整 246 条官方 benchmark 分数。

如需在不继续下载的情况下重新扫描已有缓存、转换音频并生成子集 CSV，可运行：

```bash
python benchmark/paraspeechcaps/finalize_partial_eval.py
```

缺少源音频不影响查看 benchmark 的 prompt、transcript 和标签，也不影响把同一批 prompt 用作 TTS 合成输入；但无法对照 ground-truth 音频。

## 重新生成可读文件

在项目根目录运行：

```bash
python benchmark/paraspeechcaps/prepare_preview.py
```

脚本需要 `pyarrow`，并会从官方 parquet 重新生成 JSONL、CSV、统计和 HTML 预览。

## 另一个测试集

论文还描述了 240 条 compositional evaluation prompts：12 个 intrinsic tags × 10 个 situational tags × 2 个 genders，transcript 来自 LibriTTS test。
它没有 ground-truth 语音；截至本次下载，官方数据仓库和代码仓库没有单独发布这 240 条的精确清单或生成随机种子，因此这里不能从官方文件无损复原。

数据与模型许可证为 CC BY-NC-SA 4.0；代码仓库许可证为 MIT。
