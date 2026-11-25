# FoundationPose 实时物体位姿跟踪

提供一个物体的 CAD 网格，连接 RealSense D415，即可在 RGB-D 画面中自动定位并跟踪它。支持无贴图 STL，无需手绘 mask 或点击目标。物体丢失后，程序会自动尝试重新定位。

## 开始跟踪

在项目根目录打开 PowerShell。已有环境、相机配置和网格时，直接运行：

```powershell
conda activate foundationpose
python scripts/track_live.py
```

让物体完整入画，等待窗口显示 **TRACKING**。左侧是最新相机画面，右侧是位姿对应的图像，红、绿、蓝分别表示物体的 X、Y、Z 轴。**LOST** 时让物体重新入画即可。按 **Esc** 停止并释放相机。

更换物体时，将网格放入 `data/meshes/<物体名>/`，在 [track_live.py](scripts/track_live.py) 顶部修改 `Args.mesh` 和 `Args.mesh_unit`（`mm`、`cm` 或 `m`）。默认路径为 `data/meshes/cover/cover.stl`，单位为毫米。也可以只对本次运行指定网格：

```powershell
python scripts/track_live.py --mesh data/meshes/bracket/bracket.stl --mesh_unit mm
```

## 首次安装（Windows）

准备好 NVIDIA GPU 及驱动、Conda、Git，以及带 C++ 桌面开发工具和 Windows SDK 的 Visual Studio Build Tools。以下命令会安装 Python 环境并编译本项目的 CPU 扩展。

```powershell
git clone https://github.com/ShiHao1126/FoundationPose.git
cd FoundationPose
git clone https://github.com/spatialhub-ai/spatialhub.git third_party/spatialhub
git -C third_party/spatialhub checkout --detach b4d0bcef7abddd5c872e25d92c6c10554caea3a5
conda env create -f environment.yml
conda activate foundationpose
Copy-Item configs/realsense.example.json configs/realsense.json
```

在 `configs/realsense.json` 中把 `serial` 改成 D415 的序列号，可在 RealSense Viewer 中查看。关闭 Viewer，放好自己的网格，再按上面的命令启动。首次运行会下载模型并生成该网格的模板，后续启动会复用 `cache/`。

## 查看和保存结果

每次实时运行的结果保存在 `runs/live/<运行时间>/`。

| 文件 | 用途 |
| --- | --- |
| `poses.jsonl` | 每个处理帧一条记录。读取 `valid=true` 的 `T_camera_object`，将原始网格坐标系中的米制点变换到相机坐标系。 |
| `status.json` | 当前显示状态。外部程序读取时同时检查更新时间和 `valid_now`。 |
| `timing.jsonl` | 查看帧到达和处理完成的时间，用于分析延迟与恢复耗时。 |

相机坐标为 X 向右、Y 向下、Z 向前。窗口中的平移以**毫米**显示，位姿矩阵中的平移以**米**保存。需要录下跟踪画面时运行 `python scripts/track_live.py --save_video`。

## 录制和回放 RGB-D

退出实时跟踪后，运行 `python scripts/preview_camera.py` 可录制 30 帧对齐的 RGB-D 到 `data/recordings/`。将下面的路径替换为刚录制的目录，即可离线回放：

```powershell
python scripts/track.py --sequence "data/recordings/实际录制目录"
```

回放使用 [track.py](scripts/track.py) 中的网格设置；换过物体时，同样可用 `--mesh` 和 `--mesh_unit` 指定。使用自己的序列时，目录中需包含 `cam_K.txt`（3×3 内参）、`rgb/` 和 `depth/`（同名、对齐的 PNG）。默认深度单位为毫米，其他尺度通过 `--depth_scale_m` 指定每个深度单位对应的米数。
