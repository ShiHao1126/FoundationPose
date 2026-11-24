# 自动 CAD 物体位姿跟踪

面向已知 mesh 的单个刚体，使用对齐、去畸变的 RGB-D 图像，自动生成候选 mask、初始化六自由度位姿、持续跟踪，在几何校验失败后重新全图定位。用户不需要标注 mask、点击目标或输入物体名称。内部仍使用自动 mask。

当前包含离线回放、RealSense D415 短时采集，以及持续实时采集、自动位姿跟踪与窗口显示。目标是小型无贴图打印件；厘米以内的实际定位误差、遮挡容忍度和恢复成功率需要用实际 mesh、相机及测量基准验证，不能由算法阈值推定。

## 目录与职责

```text
configs/                 共享跟踪阈值、固定模型版本
src/mesh_pose/
  contracts.py           RGB-D、位姿、质量指标、配置与后端接口
  backend.py             SpatialHub 模型加载、CAD 检测、位姿推理、渲染
  depth_renderer.py      独立持久深度渲染目标，仅回读相机 Z，复用 GPU 网格
  registration.py        保持全旋转候选和四路排序的可中断全局注册
  geometry.py            深度尺寸预筛选、局部 ICP 精修、遮挡与轮廓校验
  tracker.py             唯一的跟踪状态机及重定位策略
  sequence.py            离线数据源、JSONL 与预览视频输出
  realsense.py           RealSense 生命周期、帧对齐、重复帧过滤和单位转换
  live.py                相机与推理线程、最新帧交换、故障与退出、实时诊断记录
  viewer.py              Tk 实时画面、对应帧位姿显示、过期隐藏和停止请求
scripts/track.py          用户回放入口；Args(Tap) 集中设置默认参数
scripts/track_live.py     用户实时跟踪入口；默认连接配置中指定的相机
scripts/preview_camera.py 短时相机检查并保存可回放的 RGB-D，结束即释放设备
scripts/tests/           软件回归与真实 GPU 离线验证
third_party/spatialhub/   固定版本的上游源码，不在其中修改业务逻辑
data/meshes/<object>/     用户 mesh 统一位置，OBJ 的纹理与 MTL 同目录（不入 Git）
data/samples/             公开验证数据，保持独立（不入 Git）
data/recordings/          实际相机录制，包含内参、时间戳和设备诊断（不入 Git）
cache/                   模型下载缓存（不入 Git）
runs/                    每次运行独立目录（不覆盖历史记录）
```

输入源只生产 `RGBDFrame`，跟踪器只消费帧并返回 `TrackingResult`。实时与回放复用同一个跟踪核心。实时相机线程只负责 SDK 与最新帧，推理线程独占模型、OpenGL 和记录，主线程只负责窗口；慢初始化不阻塞相机，旧帧不排队。没有 ROS 或机器人控制。

## 本机环境

```powershell
conda activate foundationpose
cd D:\Code\FoundationPose
python -m pip install -e .
python -m pytest scripts/tests -q
```

新机器先获取固定依赖，然后创建环境：

```powershell
git clone https://github.com/spatialhub-ai/spatialhub.git third_party/spatialhub
git -C third_party/spatialhub checkout --detach b4d0bcef7abddd5c872e25d92c6c10554caea3a5
conda env create -f environment.yml
```

GPU 后端固定使用 ONNX Runtime 1.20.2 与 PyTorch 2.5.1 CUDA 12.1 的运行库；PyTorch 用于提供 CUDA/cuDNN DLL，不参与模型训练。初始化会检查四个模型实际使用 CUDA，拒绝静默退回 CPU。无需本地编译 FoundationPose CUDA 扩展。模型首次运行按 `configs/models.json` 下载指定 revision，缓存位于项目 `cache/`。

## 用户入口

持续实时跟踪（可直接修改脚本顶部的 `Args` 默认参数）：

```powershell
conda activate foundationpose
cd D:\Code\FoundationPose
python scripts/track_live.py
```

该入口默认真实连接相机。Tk 窗口左侧显示最新相机画面，右侧显示位姿对应的处理帧；右侧有推理延迟，并标出帧龄。通过校验且帧龄不超过 0.5 s 时才画出 CAD 包围框和坐标轴，并显示原始 mesh 原点的相机坐标平移（mm）及旋转矩阵。过期、丢失、故障或停止后，当前有效位姿清空。帧龄从 SDK 读取并对齐完成时的主机单调时钟计算，不代表包含全部传感器传输延迟的曝光到显示时间。

首次加载模型／模板和全局注册需要等待，初始化及重定位时先尽量保持物体静止、轮廓完整。全局注册返回的旧帧候选不会直接作为当前位姿，必须在新帧上重新通过自动 mask 与更强的初始化几何检查。已发布跟踪的输入时间间隔超过 1 s 时清除有效输出并进入恢复；该门限与界面 0.5 s 的显示有效期是不同约束。窗口的有效位姿 Hz 统计最近 5 秒实际发布数量，包含检测检查造成的停顿；单步耗时另外显示为 ms。

按“停止并释放相机”、Esc 或关闭窗口即可退出。相机线程停止采集，推理线程完成当前调用后释放 GPU／OpenGL。也可在本次运行目录创建名为 `STOP` 的空文件请求同样的正常退出；不要同时运行两个相机入口。`duration_s` 可设置有限运行时间，默认 0 表示持续运行。

实时诊断位于 `runs/live/<时间>/`：`poses.jsonl` 保存每张处理帧的历史结果，`timing.jsonl` 保存 SDK 配对、主机到达／处理完成时间及推理跳帧数；`status.json` 是带更新时间、帧龄与 `valid_now` 的显示快照，过期时矩阵为 null。历史 `valid=true` 仅表示对应输入帧通过校验，不代表现在仍有效；读取快照的外部程序也必须检查更新时间。`transitions/` 保存状态变化时的 RGB、深度与内参作为诊断证据。`preview.mp4` 只含处理帧；`live_camera.mp4` 独立于推理录制双栏显示，配套 `display_timing.jsonl` 记录视频帧号和主机时间，因此慢搜索期间也有相机证据。视频都按固定播放帧率编码，真实耗时应读取时间日志。它不是完整训练数据采集入口。

正常实时预览默认不录制视频，保存位姿、状态变化图片和诊断日志。需要复核遮挡过程时，运行 `python scripts/track_live.py --save_video`，才会写入上述两种视频及显示时间日志。

已连接设备为 D415，USB 3.2；首次使用先将 `configs/realsense.example.json` 复制为 `configs/realsense.json` 并填写相机序列号。相机设置集中在 `configs/realsense.json`。以下入口会真实打开相机，默认保存 30 帧唯一 RGB 曝光及对应的对齐深度，然后关闭相机：

```powershell
python scripts/preview_camera.py
```

它不弹出 GUI 窗口；画面保存在打印出的 `data/recordings/d415_<UTC时间>/rgb/`。检查 `camera.json` 的完成状态、重复帧／丢帧统计和时间戳差。SDK 可能用同一 RGB 帧与不同深度帧组成 frameset，适配器会跳过重复 RGB 时间戳。内参和深度单位从 SDK 读取，原始时间戳、配对诊断随录制保存。当前配置的彩色畸变系数为零，深度对齐到彩色坐标系。

短时检查不证明 RGB 与深度硬件同步或物体位姿精度。实际帧时间戳差可能接近半帧周期；快速移动需要单独验证。物体须完整处于视野内；相机初始化校验失败时不会输出有效位姿。

当前场景使用 `color_exposure=39`、`color_gain=64`：自动曝光下白色 cover 大面积饱和，降低曝光后固定目标区域的饱和比例从约 81% 降为 0%。这是当前光照下新增的工程设置，不是通用标定参数；RGB 曝光使用 SDK 原生单位，不能按深度曝光的微秒单位解释。两项同时设为 `null` 可保留打开前的设备设置；手动模式关闭时恢复原来的曝光、增益和自动曝光开关。实际设置写入 `camera.json`。成像改善不等于位姿精度或检测成功率已提高。

修改 `scripts/track.py` 顶部 `Args` 的 mesh、mesh_unit、sequence，然后运行：

```powershell
python scripts/track.py
```

当前默认模型为 `data/meshes/cover/cover.stl`，用户已确认单位毫米，外形 70 × 62 × 70 mm；移动前后的 SHA256 一致，原始 CAD 坐标未改变。更换模型时须同步确认 `mesh_unit`。默认深度每原始单位为 0.001 m。路径由脚本位置确定，不依赖当前工作目录。CLI 覆盖也可用，`python scripts/track.py --help` 查看。

离线输入格式：

```text
data/input/
  cam_K.txt              3×3 内参矩阵，空格分隔
  rgb/000000.png         uint8 RGB 文件（由 OpenCV 读取时转换 BGR→RGB）
  depth/000000.png       同名、同尺寸、对齐到 RGB 的深度；0 为无效
  rgb/000001.png
  depth/000001.png
  timestamps.json        可选：{"000000": 0.0, "000001": 0.033333}
```

RGB 与深度必须来自同一曝光时刻或已满足采集侧同步要求。匹配文件名不能证明硬件同步。没有 timestamps.json 时，仅按 fps 生成回放时间并在输出中标明。图像须去畸变；程序不猜测畸变系数或自动修改内参。

每次运行保存 `run.json`、逐帧 `poses.jsonl`、可选 `preview.mp4`，关闭时写入 `completion.json` 区分正常完成与中断／失败。若进程被强制结束、没有 completion.json，应视为未完成。状态及输出约定：

| 状态 | 含义 | valid / T_camera_object |
|---|---|---|
| SEARCHING | 尚未找到通过校验的候选 | false / null |
| CONFIRMING | 初始化连续帧确认，或周期检测漏检后的立即复核 | false / null |
| TRACKING | 本帧通过校验 | true / 4×4 矩阵 |
| LOST | 跟踪失败，尝试历史位姿局部恢复并推进全图搜索 | false / null |

`T_camera_object` 把**原始 mesh 坐标系**中的米制点变换到相机坐标系；相机 x 向右、y 向下、z 向前，平移单位米。上游内部居中产生的偏移在输出时恢复。例：毫米 mesh 顶点 `v_mm` 对应 `v_camera = R @ (v_mm * 0.001) + t`。下游必须检查 `valid`，不能继续使用上一帧矩阵替代无效结果。输出是相机坐标位姿，尚不包含相机到机器人外参。

## 恢复与校验策略

1. FastSAM 自动提出 mask，先排除缺乏有效深度或尺寸明显不符的候选，再用 DINOv2 对 mesh 多视角渲染模板进行 CNOS 匹配。检测后端保留全部达到相似度门限的 mask，不在后端截成前三个。
2. 在启动相机前可运行 `python scripts/prepare_templates.py`。程序将外观特征和 252 个朝向对应的裁剪轮廓、相对深度、投影尺寸保存在 `cache/templates/`。实时入口也可在首次运行时自动构建。启动和丢失后的全图搜索先比较各候选的尺寸、轮廓与深度，对外观排名前三的区域按几何匹配排序，先各精修 16 个候选朝向，以此筛选可能的目标区域。入选区域使用完整旋转网格判定朝向；只有通过完整配准和强校验的结果才进入新帧确认。若入选区域均未通过，则检查原外观排名前三中尚未完整搜索的区域。实时模式下，粗筛全部未命中时会在下一张新帧进行完整搜索。每个候选均与全部合格 mask 核对。
3. 初始化要求更强的几何证据：深度内点比例至少 0.8，自动 mask 与模型轮廓 IoU 至少 0.6；跟踪允许一定遮挡，使用较低的持续跟踪阈值。连续两帧通过校验才发布有效位姿。若多个空间分离的候选都通过初始化检查，则输出歧义；不提供多个相同零件之间的持久身份识别。
4. 跟踪时每帧独立进行渲染深度校验，默认每 20 帧再次做自动检测确认目标轮廓。持续跟踪及慢初始化后的新帧确认，会围绕当前预测轮廓扩展半个目标投影尺寸，在该区域自动分割并回映到整张图像；这减少小物体被全图分割漏掉的情况，不需要人工框选。首次初始化和丢失后的重新定位仍搜索全图。
5. 深度校验失败、过大运动或时间断流时立即清除有效位姿。最后可信位姿仅作为内部恢复提示：先在最新帧上局部精修，保留运动门限，并通过更强的初始化深度／mask 检查；下一张新帧再次强校验后才发布。历史位置被遮住时不向前景遮挡物精修。局部恢复失败时继续全图、完整旋转搜索，允许目标移到其他位置或大幅转动。
6. 实时全局注册在网络计算批次之间让出执行权，状态机先检查最新帧能否局部恢复；一旦恢复，关闭旧搜索。保持原有 252 个旋转候选、batch=4 的排序语义、全部候选歧义检查和五轮神经注册精修。整批裁剪准备仍有不可中断耗时，不承诺硬实时上限。离线回放在同一帧内完成搜索，复用同一个注册生成器。
7. 单次周期检测漏检立即停止发布位姿，先进入 CONFIRMING，在下一帧再次自动复核；通过则恢复输出，连续两帧复核失败才转入恢复。全局搜索失败后默认每五张处理帧重试。

初始化和恢复要求目标轮廓完整处于画面内：全图检测先排除接触图像边界的 mask，强校验还拒绝接触图像边界的模型投影。长时间遮挡实测曾暴露“仅匹配画面下沿一小块白色物体”也通过残差与 IoU 的误定位；这个额外拒绝条件防止把画面外的大部分 mesh 忽略。持续跟踪原有遮挡门限未放宽。

回退的完整全局搜索在本机约需 15 秒。若其候选到达确认阶段时目标已转动，先用当前自动 mask 重新估计中心，并在旧姿态的原有 75° 运动范围内搜索旋转候选，经过原有 80 mm 位移门限、强深度与轮廓检查后重新进入 CONFIRMING；必须再由另一张新帧强确认才发布。该路径只处理已有全局校验候选的刷新，不能替代首次全旋转搜索。强确认时局部分割没有合格 mask，会回退到整幅图像自动分割，仍使用同样的几何阈值。

可调整阈值集中在 `configs/tracker.json`。默认深度容差 8 mm、遮挡判定 15 mm、最小可见比例 30%；初始化候选点云的稳健包围盒对角线范围为 CAD 直径的 0.4–1.75 倍；ICP 最多四次，累计修正最多 20 mm／10°。这些均是初始工程设置，严重遮挡或只露出很小局部时可能无法重新初始化。

深度残差不是物体平移误差；DINO 相似度与 inlier ratio 也不是正确概率。无纹理、几何对称或接近平面的物体，以及多个相同打印件，仍可能存在不可观测方向或身份歧义。

## 验证

```powershell
python -m pytest scripts/tests -q
python scripts/tests/validate_offline.py
python scripts/tests/profile_tracking.py --mode parity
```

实时运行结束后可用 `python scripts/tests/analyze_live.py --run runs/live/<本次目录>` 汇总时序、有效位姿、丢失到恢复间隔；是否属于真实遮挡还需要结合相机视频核对。

第二条使用公开 `mustard0` RGB-D 录制，默认将 mesh 去掉贴图、导出为毫米 STL，完全不读取其 masks 目录；在真实图像序列中插入三帧合成的远处背景，检验自动初始化、无效输出与自动恢复。它是**真实模型推理 + 软件注入丢失**，图像中的实物仍是带标签的公开样例，不等同于真实无纹理打印件、真实遮挡或精度测试。公开样例下载链接见 `docs/upstream.md`。测试用例覆盖状态机、拒绝条件、全局恢复、单位转换、时间顺序和输出读取。

cover（毫米 STL）和 D415 的实际回放与实时验证记录见 `docs/realsense_validation.md`。软件测试、短序列成功、真实遮挡恢复和独立精度测量分别记录，不能互相替代。

