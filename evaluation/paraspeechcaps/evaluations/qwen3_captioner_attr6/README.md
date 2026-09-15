# Qwen3 / MiDasheng Captioner 的 ParaSpeechCaps Attr6 评测指南

本目录用于在本地可用的 ParaSpeechCaps 子集上评测 Qwen3-Captioner 的六项
语音属性。评测集包含 142 行可用标注，按相同音频去重后得到 140 个不重复
音频。它不是完整的 246 行官方测试集，因此结果只能表述为“本地可用的
140 音频子集结果”，不能表述为完整官方基准评测分数。

评测字段为：`gender`、`pitch`、`speaking_rate`、`accent`、
`intrinsic_traits` 和 `situational_traits`。集群迁移没有改变六字段规范、
提示词、真实标注归一化、按音频去重方法或评分公式；噪声字段按既定协议排除。

## 新增：Content-only Scheme A（独立协议，不覆盖 Strict Attr6）

为了降低一次六字段 JSON 输出带来的格式耦合，新增协议
`paraspeechcaps-attr6-content-scheme-a-v1`。原来的 Strict Attr6 脚本、提示词、
输出目录和报告均未修改；两套结果应分别报告，不能把 Scheme A 分数直接替换成
原 Strict Attr6 分数。

Scheme A 对每条音频分别发出 6 个独立问题：`gender`、`pitch`、
`speaking_rate`、`accent` 各返回一个选项 ID；`intrinsic_traits` 和
`situational_traits` 各返回零个或多个选项 ID，空集返回 `NONE`。选项 ID 是主解析
方式，精确的规范标签匹配仅作为确定性后备。某个字段解析失败时只把该字段记为
零分，不影响同一音频的其他五项；每个字段始终以全部 140 条音频为分母。
`field_parse_rate` 和 `all_six_fields_parse_rate` 只作为格式诊断，不作为额外加分项。

相关代码完全独立：

```text
content_scheme_a.py             # 六个问题、选项 ID 和字段级解析器
run_content_scheme_a.py         # 单候选模型推理、字段级断点续跑
score_content_scheme_a.py       # 固定分母评分和报告
../../../run_caption_content_scheme_a_8gpu.sh  # 八候选/八卡并发编排
```

八个候选及默认 GPU 映射为：

| GPU | 候选 |
| ---: | --- |
| 0 | `qwen_base` |
| 1 | `qwen8_r0` |
| 2 | `qwen8_r1` |
| 3 | `qwen8_r2` |
| 4 | `midasheng_base` |
| 5 | `midasheng4_r0` |
| 6 | `midasheng4_r1` |
| 7 | `midasheng4_r2` |

建议先在 8 卡服务器运行每模型一条音频的 smoke；八个 smoke 都成功后，再运行
完整的 140 条评测：

```bash
cd /data/L202500147/Caption/benchmark
bash run_round0_evaluations.sh caption-content-smoke-8gpu
bash run_round0_evaluations.sh caption-content-full-8gpu
```

完整评测共执行 `8 × 140 × 6 = 6720` 次字段级生成，每张 GPU 独立加载一个完整
候选模型并处理该候选的 840 次生成。重复运行相同命令会按字段断点续跑。结果位于：

```text
/data/L202500147/Caption/benchmark/round_completion_eval_runs_20260831/
  paraspeechcaps_content_scheme_a_v1/
```

如物理 GPU 编号不是 `0..7`，可按上述候选顺序覆盖映射，例如：

```bash
CONTENT_SCHEME_A_GPUS=7,6,5,4,3,2,1,0 \
  bash run_round0_evaluations.sh caption-content-full-8gpu
```

## 0. 本轮最简单的执行方法

原 Strict Attr6 本轮入口已经支持以下七种待比较对象，不需要合并 LoRA，也不能把 LoRA 目录
冒充完整模型目录：

| 名称 | 基础模型 | checkpoint |
| --- | --- | --- |
| `qwen8_r0` | Qwen3-Captioner | 8 卡正式训练第一轮 `round_000/caption_final` |
| `qwen8_r1` | Qwen3-Captioner | 8 卡正式训练第二轮 `round_001/caption_final` |
| `qwen8_r2` | Qwen3-Captioner | 8 卡正式训练第三轮 `round_002/caption_final` |
| `midasheng_base` | MiDasheng | 无适配器 |
| `midasheng4_r0` | MiDasheng | 4 卡正式训练第一轮 `round_000/caption_final` |
| `midasheng4_r1` | MiDasheng | 4 卡正式训练第二轮 `round_001/caption_final` |
| `midasheng4_r2` | MiDasheng | 4 卡正式训练第三轮 `round_002/caption_final` |

8 卡 Qwen 和 4 卡 MiDasheng 的三轮训练均已产生 `round_000..002/commit.json`、
`latest.json` 和 final checkpoint。前一批四个对象已经完成 140 条本地子集评测；
补充入口会在三张 H100 上并行运行尚未评测的 Qwen r2、MiDasheng r1/r2，默认
分别使用物理 GPU 0、1、2，使用互不混用的新输出目录并保留断点续跑记录。可用
`CAPTION_GPU_QWEN8_R2`、`CAPTION_GPU_MIDASHENG4_R1`、
`CAPTION_GPU_MIDASHENG4_R2` 覆盖 GPU 分配，三个值必须不同。
先完成 TTS README 第 0 节的权限修复命令，之后执行：

```bash
cd /data/L202500147/Caption/benchmark

bash run_round0_evaluations.sh check
bash run_round0_evaluations.sh caption-remaining-smoke
```

必须看到 `caption-remaining-smoke` 的三个 case 都以 `[DONE]` 结束。所有新增
smoke 都成功后，再执行：

```bash
bash run_round0_evaluations.sh caption-remaining-full
```

重复执行同一条命令会从各自目录续跑；不要改脚本中的目录名。
原有四个和新增三个完整报告分别位于：

```text
paraspeechcaps/evaluations/qwen3_captioner_attr6/runs/eval_qwen8_round0_full_20260830_run01/reports/final_report.md
paraspeechcaps/evaluations/qwen3_captioner_attr6/runs/eval_qwen8_round1_full_20260830_run01/reports/final_report.md
round_completion_eval_runs_20260831/paraspeechcaps/eval_qwen8_round2_full_20260831_run01/reports/final_report.md
paraspeechcaps/evaluations/qwen3_captioner_attr6/runs/eval_midasheng_base_full_20260830_run01/reports/final_report.md
paraspeechcaps/evaluations/qwen3_captioner_attr6/runs/eval_midasheng4_round0_full_20260830_run01/reports/final_report.md
round_completion_eval_runs_20260831/paraspeechcaps/eval_midasheng4_round1_full_20260831_run01/reports/final_report.md
round_completion_eval_runs_20260831/paraspeechcaps/eval_midasheng4_round2_full_20260831_run01/reports/final_report.md
```

新增三个结果放在 benchmark 根目录下的共享可写目录，是为了允许另一项目容器
直接运行 GPU 推理；旧的四个结果仍保留在原 `paraspeechcaps/.../runs/` 中。

这里的 MiDasheng base 是 Captioner，只评 ParaSpeechCaps；它不是 TTS 模型，
因此不存在“拿 MiDasheng base 生成 EmergentTTS 音频”这项评测。两个正式分支
每一轮的 TTS checkpoint 则按另一份 README 评 EmergentTTS-Eval。

## 1. 真实标注、音频和本地清单

真实标注和测试音频已经存在，不需要模型重新生成：

- 可用真实标注：`../../data/test_available.csv`
- 原始完整标注：`../../data/test.csv`
- 测试音频：`../../audio/test/{benchmark_index:03d}.wav`
- 已整理并验证的 140 条本地清单：`runs/new_cluster_default/manifest.jsonl`
- 清单元数据：`runs/new_cluster_default/manifest_meta.json`

这里的清单文件只是推理和打分程序使用的本地索引。它把已有真实标注、
140 个不重复音频、样本 ID 和本地音频路径整理到同一份 JSONL 中，
不会调用模型、不会重新标注，也不会生成新的真实标注。

当前清单已经生成并验证过。只要真实标注和音频位置没有变化，
正常评测应直接复用它，不需要再次运行 `prepare_manifest.py`。

只有在原始标注或音频位置确实发生变化时，才重新生成默认清单：

```bash
cd /data/L202500147/Caption/benchmark/paraspeechcaps/evaluations/qwen3_captioner_attr6

/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python \
  prepare_manifest.py
```

## 2. 本地模型和环境

- Captioner 基础模型：`/data/L202500147/Caption/models/Qwen3-Omni-30B-A3B-Captioner`
- Captioner Python：`/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python`
- MiDasheng 基础模型：`/data/L202500147/Caption/models/MiDashengLM-7B-1021-BF16`
- MiDasheng Python：`/data/L202500147/miniconda3/envs/midasheng-captioner/bin/python`
- 默认清单：`runs/new_cluster_default/manifest.jsonl`

可以通过 `--model-dir` 或 `CAPTIONER_MODEL_PATH` 覆盖模型路径，通过
`--python` 或 `CAPTIONER_PYTHON` 覆盖推理子进程使用的 Python。

当前加载器已经支持“完整基础模型 + 可选 PEFT LoRA”：Qwen 适配器严格挂载
到 `Qwen3OmniMoeForConditionalGeneration.thinker`，MiDasheng 使用自己的
`AutoProcessor`、chat template 和 tokenizer，并把适配器挂载到原生
`AutoModelForCausalLM`。加载前会校验 adapter config 中的基础模型路径、文件
可读性和 SHA-256；加载后会确认确实存在被冻结的 `default` LoRA 参数。

手工运行训练后模型时，`--model-dir` 始终指向完整基础模型，另加
`--adapter-dir /path/to/round_NNN/checkpoints/caption_final`。MiDasheng 应使用
`run_midasheng_captioner.py`，不要用 Qwen processor 处理 MiDasheng 音频。

迁移来的根级 `manifest.jsonl`、`manifest_meta.json`、`outputs/`、`reports/`
和 `runtime/` 是旧服务器的历史证据，不得覆盖。所有新评测都应写入
`runs/` 下的新目录。

## 3. 推荐方式：推理和打分分开执行

以下示例评测默认的 Qwen3-Captioner 基础模型，并复用已经验证的 ground-truth
清单。先进入目录并设置本次运行：

```bash
cd /data/L202500147/Caption/benchmark/paraspeechcaps/evaluations/qwen3_captioner_attr6

PSC_PY=/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python
PSC_MANIFEST="$PWD/runs/new_cluster_default/manifest.jsonl"
PSC_RUN="$PWD/runs/base_qwen3_captioner_20260830_run01"
mkdir -p "$PSC_RUN"
```

### 3.1 对 140 个音频执行 Captioner 推理

单卡执行：

```bash
"$PSC_PY" run_qwen3_captioner.py \
  --manifest "$PSC_MANIFEST" \
  --output-dir "$PSC_RUN/outputs" \
  --gpu-list 0 \
  --resume
```

如果有两张可见且空闲的 GPU，可以改为：

```bash
"$PSC_PY" run_qwen3_captioner.py \
  --manifest "$PSC_MANIFEST" \
  --output-dir "$PSC_RUN/outputs" \
  --gpu-list 0,1 \
  --resume
```

这里是数据并行：每张 GPU 分别加载一套完整 Captioner，再各自处理一部分音频；
不是张量并行。当前入口最多使用两张 GPU。

`--resume` 会从分片 JSONL 中读取已经成功的样本，只处理失败或缺失样本。
重新执行时必须保持模型、清单、输出目录和生成参数不变。推理完成后检查：

```text
$PSC_RUN/outputs/predictions.jsonl
$PSC_RUN/outputs/run_metadata.json
$PSC_RUN/outputs/shards/
```

`run_metadata.json` 中应有：

```text
selected_samples = 140
merged_selected_records = 140
successes = 140
```

如果 `successes` 少于 140，可以先重跑相同推理命令；仍然失败的记录会在最终
评分中按零分处理，并反映在覆盖率中，不能隐瞒。

### 3.2 使用已有真实标注打分

推理结束后，用同一份清单中的真实标注对预测进行评分：

```bash
"$PSC_PY" score.py \
  --manifest "$PSC_MANIFEST" \
  --predictions "$PSC_RUN/outputs/predictions.jsonl" \
  --output-dir "$PSC_RUN/reports"
```

该阶段不需要 GPU。最终结果位于：

```text
$PSC_RUN/reports/
├── final_report.md
├── summary.json
├── per_sample.csv
├── per_source.csv
└── per_label.csv
```

`final_report.md` 是便于阅读的总报告；`summary.json` 保存总分、六字段分数、
覆盖率、自助法置信区间和各来源结果；`per_sample.csv` 可审计每条样本。

## 4. 现有一键入口

本目录已有一个按顺序执行“断点续跑推理 → 打分”的入口：

```bash
cd /data/L202500147/Caption/benchmark/paraspeechcaps/evaluations/qwen3_captioner_attr6

/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python run_full.py
```

它固定使用默认基础模型、`runs/new_cluster_default/manifest.jsonl`，并把结果
写入 `runs/new_cluster_default/outputs/` 和 `runs/new_cluster_default/reports/`。

这个入口适合只评测一次默认基础模型。比较多个模型时，不建议使用一键入口，
因为它不能为每个模型指定独立运行目录；应使用第 3 节的两阶段命令，防止
`--resume` 错误复用另一个模型的历史预测。

## 5. 单样本冒烟测试

单样本测试必须使用独立目录，不能写入完整评测目录。先从已验证的清单
中取一条记录，真实标注仍然来自原始基准评测：

```bash
cd /data/L202500147/Caption/benchmark/paraspeechcaps/evaluations/qwen3_captioner_attr6

PSC_PY=/data/L202500147/miniconda3/envs/qwen3-captioner/bin/python
PSC_SMOKE="$PWD/runs/smoke_qwen3_captioner_20260830_run01"
mkdir -p "$PSC_SMOKE"

sed -n '1p' runs/new_cluster_default/manifest.jsonl > "$PSC_SMOKE/manifest.jsonl"

"$PSC_PY" run_qwen3_captioner.py \
  --manifest "$PSC_SMOKE/manifest.jsonl" \
  --output-dir "$PSC_SMOKE/outputs" \
  --gpu-list 0 \
  --resume

"$PSC_PY" score_smoke.py \
  --manifest "$PSC_SMOKE/manifest.jsonl" \
  --predictions "$PSC_SMOKE/outputs/predictions.jsonl" \
  --output-dir "$PSC_SMOKE/score"
```

单样本结果位于：

```text
$PSC_SMOKE/score/summary.json
$PSC_SMOKE/score/scored_sample.json
```

不要用完整评测的 `score.py` 给单样本打分，因为它要求 140 条清单记录，
并会把其余缺失样本计为零分。单样本必须使用 `score_smoke.py`。

## 6. 禁止事项和常见错误

- 真实标注已存在；不要把 `prepare_manifest.py` 理解为重新生成标注。
- 正常运行直接复用 `runs/new_cluster_default/manifest.jsonl`。
- 不要覆盖迁移来的根级历史 `outputs/`、`reports/` 和 `runtime/`。
- 不要让不同模型共用同一个新运行目录或分片文件。
- 不要把 LoRA 适配器目录直接传给 `--model-dir`。
- 不要用 Qwen Python/processor 运行 MiDasheng；使用 `run_midasheng_captioner.py`。
- 该本地子集是 140 个不重复音频，不是完整 246 行官方测试集。
- 失败、无效或缺失预测会留在 140 条分母中并按零分计算。
