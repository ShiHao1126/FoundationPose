# 上游来源与实现边界

| 项目 | 本项目使用方式 |
|---|---|
| [SpatialHub](https://github.com/spatialhub-ai/spatialhub) | 直接调用 CNOS、FastSAM、DINOv2、FoundationPose ONNX 和 ModernGL 渲染；固定提交 b4d0bcef7abddd5c872e25d92c6c10554caea3a5 |
| [STORM](https://github.com/YuDeng321/STORM) | 参考跟踪验证失败后重新初始化的组织方式；未引入 TOM 权重、参考图重建网络或其精度结论 |
| [foundationpose-ros2-tracker](https://github.com/ismael-esp/foundationpose-ros2-tracker) | 参考持续检测校验与重定位流程；未引入 ROS2、文字检测、方向锁定及机器人外参 |

本项目不是把三个仓库全部拼装运行。直接推理依赖集中在 SpatialHub 适配器，恢复策略由本地独立状态机实现，便于单元测试与更换检测器。没有修改上游源码，也没有复制后两者的模型实现。

本地 `registration.py` 的候选精修更新与四路淘汰排序来自上述固定提交的 `models/foundationpose/adapter.py`，按 Apache-2.0 使用；仅增加协作式 yield，保持整批裁剪、原网络批量和候选顺序。专用 `depth_renderer.py` 使用相同投影、裁剪和相机 Z 定义，复用上游网格缓冲区，在其释放前关闭。对照验证入口为 `scripts/tests/profile_tracking.py --mode parity`。

SpatialHub 代码为 Apache-2.0；下载的模型保留其各自上游许可，不能把适配层许可视为所有权重的许可。模型文件及固定 Hugging Face revision 位于 `configs/models.json`。

公开回放数据来自 SpatialHub FoundationPose 示例引用的 [mustard0.zip](https://drive.google.com/file/d/1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3/view)，解压至 `data/samples/mustard0/`。只用 rgb、depth、cam_K.txt、mesh；masks 不参与初始化或运行。公开数据不含用于本次验收的独立计量真值。

CUDA 运行库选择依据 [ONNX Runtime CUDA EP 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。当前机器 NVIDIA 驱动 555.97，未升级驱动或安装系统 CUDA Toolkit。
