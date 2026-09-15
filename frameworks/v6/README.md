# V6 · AudioOnly4Attr

四属性加转录的 audio-only 实现。只更新 Captioner GRPO 与 TTS SFT。

先读 [训练方法](../../docs/TRAINING.md) 和 [代码索引](../../docs/CODE_MAP.md)。基础入口为 `scripts/run_v6.sh`，实际使用的 API 并发与持续重试扩展为 `run_v6_api_workers.sh`。

配置仍保留原始绝对路径。模型、音频、参考数据与运行缓存均不在仓库中，部署前参照 [运行依赖](../../docs/DEPENDENCIES.md)。当前原 run 已完成十轮；这个目录是源码导出，不是旧 run 的可恢复副本。
