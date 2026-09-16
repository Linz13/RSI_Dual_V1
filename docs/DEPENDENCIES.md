# 运行依赖与源码边界

此仓库用于代码和实验的检查、讨论与追溯，未把整个服务器环境搬进仓库。源码采用复制快照，保留原实现和配置路径；没有在导出过程中修改训练奖励、loss、模型或数据。

## 外部资源

| 资源 | 用途 | 仓库保存内容 |
|---|---|---|
| MiDashengLM-7B-1021-BF16 | Captioner 基座 | worker、配置、adapter 元数据 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign / Tokenizer | TTS 与 codec | worker、配置、SFT/GRPO 实现 |
| Qwen3-ASR-1.7B | 转录与 V6 内容检查 | 调用代码与配置 |
| emotion2vec-plus-large | 情绪打标 | 本地专家调用代码 |
| V5 其他本地打标专家 | 语速、口音等 | labeling 适配器及依赖约束 |
| Qwen / Gemini / GPT API | 固定打标或文本评审 | 请求实现、模型配置；不含密钥 |
| 原始训练池与参考转录 | 训练数据 | 数据统计、配置路径；不含完整标签集或音频 |
| Benchmark 数据与指标模型 | 独立评测 | 下载/准备/推理/评分脚本；不含数据包或指标权重 |

模型名字并非不可变版本标识。以原配置、评测 inventory、checkpoint commit 的哈希记录为准；本次没有重新下载或对所有大权重重算哈希。

## 在其他路径运行前

1. 分别准备 Captioner、TTS、本地专家、评测所需的 Python 环境。V5 的 `pyproject.toml` 和 setup 脚本、`labeling/environment/`、各 benchmark README 提供依赖线索；它们不是跨所有 GPU/系统保证可复现的单一锁文件。
2. 配置模型、音频、转录、数据 manifest 和 Python 可执行文件的路径。V5 有相对路径继承，V6 主要是绝对路径；评测脚本也有指向原目录的默认值。
3. 将 labeling 的 `source_root` 指向本仓库 `labeling/source`，按 `labeling/configs/config.target.example.json` 构建本地配置。第三方模型包仍需单独安装。
4. 通过环境变量提供密钥。训练打标使用 `QWEN_API_KEY` / `GEMINI_API_KEY`；V5 文本 judge 支持 `V5_JUDGE_URL` / `V5_JUDGE_KEY`。DSD 旧评分脚本还会读取服务器上的私有 `api/gemini_audio.py`：仓库没有复制该含密钥文件，需要提供本地私有配置或适配为环境变量读取。
5. 先对新目录执行 CPU check，再在 GPU 节点做小规模验证。不要从导出目录直接 `resume` 原历史训练路径；身份校验涉及源码、外部依赖、路径和缓存，不是只拷贝 adapter 就能恢复。

服务器原路径仍出现在代码、配置、历史报告和 provenance 中，用于追溯；它们不是 GitHub 下载链接。源目录和导出目录之间的映射见 `provenance/files.json`，独立评分实验另见 `provenance/bidirectional_scoring.json`。

## 独立双向评分实验

新增 [双向评分 pilot](../experiments/bidirectional_scoring/README.md) 只使用 MiDashengLM 和 Qwen3-TTS 基础模型，没有训练或付费 judge。`tools/verify_bidirectional_results.py` 在克隆内只用标准库复算已保存分数；不要求上述模型或音频。

`experiments/bidirectional_scoring/src/` 保留运行时原源码及相对目录假设，未为导出修改评分逻辑。真实评分需恢复原外部资源布局，或显式配置源码中的数据、模型、Python 环境路径；不能把仓库中的结果目录直接当作完整可播放的 run。该 pilot 的最终人工标签、小规模转写及候选条件已收录，完整音频和数据集未收录。

## 不包含的内容

模型权重、优化器状态、全量 WAV、原始数据集、虚拟环境、API 原始响应缓存和日志不在导出范围。两份试听 HTML 内嵌了较大的音频数据，因此只保留统计与 listening manifest；在网页版阅读这些 manifest 不能自动访问服务器音频。

`experiments/v6/examples/` 每轮取原 rollout 文件前两组，保留文字和轨迹结构，用于理解字段，不是随机评测样本。原始报告中引用的完整音频或其他未收录文件仍需回服务器查看。

历史配置、测试和诊断脚本可能依赖原目录，仓库检查只证明文本文件、来源哈希与语法完整，不代表在新路径通过了所有 GPU/API/数据集测试。

## 第三方代码

`evaluation/InstructTTSEval-public/eval/` 包含原有上游评审脚本；labeling 和各 benchmark 也引用外部项目。保留原文件署名与已有许可证，未替外部项目重新指定许可证；发布或复用时按对应项目条款处理。来源记录见 `provenance/labeling/` 和各 benchmark README。数据集与模型不在本仓库分发。
