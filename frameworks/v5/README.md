# V5

MiDasheng Captioner 与 Qwen3-TTS VoiceDesign 的完整双循环实现，源码来自 FastResume 修订。

先读 [训练方法](../../docs/TRAINING.md) 和 [代码索引](../../docs/CODE_MAP.md)。训练入口为 `scripts/run_v5.sh` / `scripts/v5_launcher.py`；主配置为 `configs/v5_midasheng.yaml`。

源码、依赖配置、测试保留原实现，包括继承的旧版本辅助文件。只有一套 V5 代码；LabelRobust / FastResume 的历史关系见仓库首页。

该导出没有模型、训练数据、旧 run 缓存和原私有 API 示例。配置仍带原服务器路径，先阅读 [运行依赖](../../docs/DEPENDENCIES.md)，不要直接对原 run 执行迁移或恢复脚本。
