# 最小训练与验证协议

## 环境与离线约束

固定环境：Python 3.12、Ultralytics **8.4.189**、Torch **2.7.1 CPU**、torchvision **0.22.1 CPU**。训练使用独立环境，不修改生产依赖；必须先核对实际解释器、包版本及 CPU build。requirements 只 pin 直接依赖；另在私有输出记录 `pip freeze`、Python/平台/设备、脚本与参数版本。升级包需重新测 callback/head，不能宽松写成 `>=8.4`。

CLI 在 import 前设置任务独立 YOLO_CONFIG_DIR、YOLO_AUTOINSTALL=false；关闭 sync、clearml、comet、dvc、mlflow、raytune、tensorboard、wandb，屏蔽固定版本的 integration callback 注册（包含 events/platform），避免字体自动下载。Python socket guard 在 import 前阻断 DNS/connect/sendto；它只作纵深防护，**不是对恶意 checkpoint 或 native 代码的沙箱**。请在操作系统无外连环境运行，依赖、字体、权重均预先人工准备。没有 HUB 登录、上传 API、artifact 上传或云 fallback。

版本固定的内部适配必须在目标机核对；一旦 upstream 调用位置变化，不得静默恢复网络。允许的超参数白名单不包含 `resume/model/data/project/cfg/pretrained` 等能改变来源或触发下载的选项。workers=0、单设备，避免 DDP/子进程绕过进程内约束。其余 recipe 可通过 JSON 自行调节。

## 输入契约

所有真实输入、输出位于 Git checkout 外；输出目录必须全新，且不在 dataset 根内。真实数据没有“临时可提交”例外。示例 hparams 须复制到私有目录后使用。

`data.yaml` 仅允许 `path/train/val/test/names/nc`，针对标准 detection YOLO 数据集：

- `path` 是已存在的本机绝对目录；不接收网络、压缩包、download 脚本、NDJSON、Platform URI。
- `train`、`val` 必填；`test` 可选。值为根目录下已有目录/`.txt` 图片列表，或其列表。
- `.txt` 中图片须为已存在的绝对路径，或按上游语义以 `./` 相对该列表所在目录；不能含 URL。所有引用均必须在 dataset 根内。
- 数据采用标准 `images/` 与 `labels/` 平行布局，类名/ID由用户定义；没有内嵌任何样本或标签。
- `--weights` 必须是可信、已有 `.pt`。训练时为对应官方 YOLO11n 或 YOLO26n；验证时为已选 checkpoint。由用户从官方渠道准备并私下核验来源/许可证；入口不下载。
- `--family` 记录用户声明，不认证模型来源与 nano 架构。正式运行前须核对模型 YAML/scale、task/head、参数量与官方来源；不能只按文件名判断。

`--device cpu` 为默认；其它单设备值由固定 Torch/Ultralytics 是否支持决定，不自动改用 CUDA。A770 不是 CUDA，未验收的 XPU/DML 不能宣传支持。

## 分组、冻结与防泄漏

先定义事件/时间段，再人工赋 `group` 与 `split`。私有 manifest 为 JSON array，每条至少 `id/group/split`；可含用户提供的内容 `sha256` 以拦截跨 split 完全重复。允许 train/val/test/challenge/quarantine。审计器只检查、不重新分配或复制数据。

**事件跨 split 是错误；近重复即使 hash 不同也必须同组。** 按采集会话、床移动、连续视频段/邻近时间分组，边界留隔离缓冲。hash 相等检查不识别近重复，程序不能证明时间独立或检查 YAML 与 manifest 一一对应，需人工核对。固定摄像头背景相似不是跨环境泛化证据。

准备私有 JSON 文件路径数组给 `freeze.py create`：包含原始 data.yaml、hparams、官方权重、group manifest、标注规则、阈值协议、源脚本，以及 **dataset 根内每个原始文件**（图片、标签、split 清单等）。路径均为绝对本地文件；不包含输出。生成 freeze 本身含路径/hash，必须留在 Git 外。

训练入口要求 freeze 覆盖 YAML、hparams、权重和整个数据根的文件；校验内容后启动，结束再次校验。上游会写标签 `.cache`，因此排除 `.cache`；`cache=False` 不表示数据目录完全只读。将数据复制到独立实验目录，不对生产目录直接训练。数据根不支持 symlink，不要混入无关私有资料或旧日志。不得在训练中编辑数据；开始/结束 hash 不是防并发修改的原子快照，也不能侦测中途改动后又恢复，需外部冻结/权限纪律。

生成 freeze 不生成时间戳、样本 ID、样本 hash 的公开例子；不把本项目真实 freeze 当复现资产。

## 微调、选模、阈值、独立验证

1. 在冻结 train 上训练、val 上选 checkpoint；显式 `nms=True`，YOLO26 采用 one-to-many+NMS 路径。loss 回调接收 dict/tensor、检查 finite，epoch 进度截到实际总数，避免最终 val 再次回调越界。上游 lazy backend 设置可能重置线程数，因此训练/验证开始回调会重设预算，进度记录实际线程和设备。CSV/真实 epoch 是训练证据，进度文件不是唯一依据。
2. 先做少量 epoch 的功能性 smoke，只用 train/val；保存重载与验证真实 head、callback 类型。正式训练必须从原始官方预训练权重重新开始，不续接 smoke。
3. 在第一次 test 前，用 val 上预注册的阈值候选、IoU 匹配规则、top1 策略和错误目标惩罚选阈值。`common.select_threshold` 只接收预计算的 val 计数，按 `(TP-FP_boxes)/n`，同分取较高阈值；调用方须正确计算计数。这不是完整评估器，也不自动生成 test 预测。
4. 写私有阈值协议，记录权重/hash、配置、NMS/IoU/max_det、最高置信度排序、padding/clip、分辨率、业务成功门槛。冻结后不得依据 test 修改阈值。
5. 新建评估 freeze，把选中的 best.pt 加入；使用新的输出目录，`validate --split test` 仅进行独立数值验证。Ultralytics 返回的是检测指标，**不是实际 top1 裁切成功率或 Qwen 准确率**。CLI 不会读取阈值文件或自动完成业务级 crop 评分；该阶段需独立评估脚本/人工复核，不要把默认 val conf 当已选部署阈值。
6. challenge 单独留存，不在本 CLI 自动混入 test。每个挑战集若反复用于开发，就已成为已知基准；另收新隔离事件才可声称新盲测。

## CLI 与合成测试

先将示例超参数复制到仓库外的私有目录。以下变量由用户指定，权重与数据均须提前准备；`PRIVATE_ROOT` 是仓库外的绝对路径，输出目录必须不存在：

```sh
python training/audit_groups.py "$PRIVATE_ROOT/groups.json"
python training/freeze.py create --files-from "$PRIVATE_ROOT/files.json" --output "$PRIVATE_ROOT/freeze.json"
python training/train.py train --family yolo26n --data "$PRIVATE_ROOT/data.yaml" --weights "$PRIVATE_ROOT/pretrained.pt" --hparams "$PRIVATE_ROOT/hparams.json" --freeze "$PRIVATE_ROOT/freeze.json" --output "$PRIVATE_ROOT/new-run"
# 独立评估前重新冻结，加入已选 best.pt；不使用 test 选阈值。
python training/train.py validate --family yolo26n --data "$PRIVATE_ROOT/data.yaml" --weights "$PRIVATE_ROOT/best.pt" --hparams "$PRIVATE_ROOT/hparams.json" --freeze "$PRIVATE_ROOT/eval-freeze.json" --output "$PRIVATE_ROOT/new-eval" --split test
```

YOLO11n 使用相同入口与对应权重，改 `--family yolo11n`。实际运行/测试在选定目标机器的隔离环境执行；编辑机器不承担真实图像处理。mock 测试仅需 Python 标准库，不下载模型或训练图：

```sh
python -m unittest discover -s training/tests -p 'test_*.py' -v
```

mock 不是模型验收。两个家族都需另做 train/val-only smoke、保存重载和 val 预测，核对实际 CPU/head/callback；1 epoch 功能性 smoke 不代表检测精度。可复用通用流程，不复用私有训练报告的样本数量、事件、时间、hash 或成功率。

## 评估与隐私产物

分别报告：可评分 recall/IoU、漏检、错目标、FP/重复框、不可评分数，以及全帧保守分母。无输出按漏检，不用 GT/上一帧补 crop；不可评分不伪填成功。图像后处理复核 GT overlay、预测 overlay、真实 padding/clip crop（含 misses），检查孩子可见区域是否被截断。应由独立标注者/复核者完成，Agent 自审不是独立人类验收。

测速须写清：冷启动/热机、输入分辨率、batch/device/线程、decode/preprocess/inference/NMS/crop/JPEG/network/Qwen 边界。重复测速不是新增独立样本。YOLO26 官方 NMS-free ONNX 速度不能与这里的 OTM PyTorch 精度混称。

输出含 args、settings、checkpoints、指标、协议、路径、错误栈，**全部是私有数据**。stdout/stderr 也可能含路径、类名或服务地址。只发布抽象方法与合成示例，不直接拷贝日志、CSV 或 dashboard 截图。入口无自动清理，所有者另定存储和保留策略。
