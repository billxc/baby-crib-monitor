# Baby Crib Monitor

本地优先的婴儿床占用观察项目。包含**纯前端画面裁切工具**、**离线历史画面单图判断工具**和**60 秒串行实时观察 runner**。

## 当前可用：裁切工具

直接用浏览器打开 [tools/crop/index.html](tools/crop/index.html)，无需后端、构建或模型服务。

1. 点击「打开图片」选择自己的图片。默认画面是合成示意图，不含真实摄像头数据。
2. 鼠标或触摸拖动框选，点击「裁切选区」。
3. 输入角度，正数为顺时针，点击「应用旋转」。可重复旋转、裁切，或撤销。
4. 导出 PNG 或 JSON。JSON 保存按顺序执行的操作，发给后续处理程序即可。

图片处理和导出全部在浏览器内完成，不上传文件、不调用第三方服务，不自动保存画面。

## 参数约定

JSON 包含 `source_size`、顺序执行的 `operations` 和 `output_size`。

- `crop`：`x/y/width/height` 为**上一步输出图像**的像素坐标，原点在左上角。
- `rotate`：`clockwise_degrees` 为顺时针角度，围绕中心旋转；`expand: true` 扩展画布，`fill` 为扩展区域颜色。
- 未点击应用的旋转、当前有效选区，也会纳入导出的结果。
- `source` 仅保存文件名，不包含图片内容；分享参数前仍请检查文件名。

## 源码与私有数据

| 可进入 Git | 只能留在本机 |
| --- | --- |
| 通用前端、合成示意图、合成数据测试、文档 | 家庭画面、录像、裁切图片、模型输入输出、实际坐标参数 |
| 通用工具及占位配置 | 摄像头地址/凭据、Cookie、API Key、实际服务地址、部署记录和本机发布脚本 |

建议将私有数据放在**仓库外的独立目录**；临时需要在仓库内使用时，只放到已忽略的 `.local/`。不要用真实画面当测试夹具，不把含图片的 data URL 写进源码。仓库初始历史从无家庭数据的干净代码创建。

发布前检查 `git status --short`、`git diff --cached` 和 `git ls-files`；不要对本机数据目录执行 `git add -f`。

## 开发测试

前端测试需要 Node.js 20 或更新版本；直接打开裁切工具无需 Node.js：

```sh
npm ci --include=dev
npx playwright install chromium
npm test
```

测试使用合成示意图，Chromium 沙箱开启。无需真实摄像头或任何账号。网站部署时只需复制 `tools/crop/` 中的两个静态文件；现有反代/登录保护由部署环境负责，不属于此工具。

## 离线历史数据运行器

`runtime/runner.py` 在运行机器上下载已选录像片段、定点抽帧、按导出的参数旋转裁切，然后向 OpenAI 兼容视觉接口发送**一张待判断图**。不使用参考图。前端工具仍无需后端。两个 runner 均需要 Python 3.10 或更新版本及以下依赖。

```sh
python -m pip install -r runtime/requirements.txt
python -m unittest discover -s runtime -p 'test_*.py'
python runtime/runner.py /path/to/private/config.json
```

配置和数据必须放在仓库外。配置字段：`manifest`、`crop_config`（相对配置文件路径），`output_dir`、`model_base_url`、`model`、`interval_seconds`。采样清单由部署环境提前按固定间隔生成，每项包括 `id`、`epoch`、`timestamp`、`video_url`、`offset_seconds`；没有录像时 `video_url` 为 null，不拿别的时刻代替。录像与模型地址不写入源码，HTTPS 保持正常证书校验。

每帧输出原始 `yes / no / unclear` 及滤波结果。默认使用最近3次采样的多数决：至少2票才切换；`unclear` 不投票，平票或票数不足时保留上一状态；第一次明确结果用于初始化。缺失录像标记 `missing` 并重置滤波，不跨录像缺口延续状态。这是因果滤波，不使用未来帧、不回填转换时间。

输出包括 `frames/` 裁切图、可断点续跑的 `results.jsonl`、`progress.json`，完成后生成 `results.csv`、`summary.json`、`segments.json`。输出保留原始不明确/缺失标记，不能只看滤波结果推断这些时段都被成功观察。有录像索引但目标时刻没有可解码视频帧时，记为 `missing`，并记录 `missing_reason: no_frame_at_sample_time`；不移动采样时刻、不用别的画面替代。重跑必须使用相同冻结配置和采样清单。

尚未实现自动发现录像或在线通知。本项目不做活体检测，也不把床位占用信号解释为已确认睡眠。

## 实时观察（静态 ROI）

复制 `runtime/live.example.json` 到仓库外并填写私有配置，再将裁切工具导出的 JSON 放到其相邻目录。运行：

```sh
python runtime/live.py /path/to/private/config.json --cycles 3
# 不加 --cycles 则持续运行；有限验证同样等待正常的 60 秒间隔
python runtime/live.py /path/to/private/config.json
```

仅从运行机器通过 RTSP TCP 抽取一帧，完整解码后按相同的有序变换裁切，再发起一个请求（仅含这一张裁切图）。复用历史 runner 的 prompt、JSON schema、temperature=0、seed=42、max_completion_tokens=32、enable_thinking=false。不会自动切换模型或回退到云服务。两个 runner 都只向私有配置中的 `model_base_url` 发送裁切图；示例使用不可用的 `example.invalid` 占位地址，不内置实际服务。请自行运行并选择可信的本地模型端点；若改为远程或云端地址，图片就会离开运行机器。`ffmpeg` 可在私有配置中指定，否则使用 imageio-ffmpeg 内置路径。

- 固定 **60 秒**周期，串行、不重叠；取流和推理有独立超时。耗时过长则跳过错过的 tick，不补跑、不积压；跳过后重置滤波并在当前状态和下一条记录中标记缺口。
- 因果最近 **3 次观测**多数决（不是旧的 120 秒采样）：`unclear` 不投票，至少 2 票才切换，票数不足/平票保持；首次明确结果初始化。图像缺失标为 `missing` 并清空状态。推理失败但图像存在时标为 `unclear`，另记错误，下一周期继续。
- 输出目录持有 OS 文件锁，避免两个实例同时采样；异常退出自动释放。每次启动创建独立 session、重新初始化滤波，不跨进程停机延续票数。
- `sessions/<unique>/run-config.json` 冻结配置，`full/` 保存原图，`frames/` 保存裁切图，`results.jsonl` 逐行保留 UTC 开始/完成时间、raw、filtered、unknown 保持标记、错误、缺失原因、模型响应及图像尺寸。可据此汇总时段，**不得将缺失/unknown 的保持状态视为观察成功**。
- 根目录 `current.json` 原子更新当前结果；`heartbeat.json` 标记 PID、session、采样/等待/有限验证完成阶段。等待时每 5 秒更新，采样阶段受超时限制；调用方应同时检查心跳年龄和记录时间，不能把文件存在当作健康。
- 所有输出（含模型响应、画面、冻结配置及错误详情）均为私有数据；标准输出、标准错误和任务调度器日志也可能包含私有地址或凭据，不可直接贴到公开 issue 或提交到 Git。`output_dir` 请填写仓库外的绝对路径；输入配置和输出目录应仅允许所有者访问。runner 不自动删除历史产物；持续保留原图/裁切图会增长磁盘占用，需由所有者另行制定保留策略。
- 当前明确标为 `roi_mode: static`。摄像头转动后固定裁切可能不再对准床区；尚无动态床区识别或可靠性保证，不可用于婴儿安全告警。

常驻运行可使用系统任务调度器：绝对 Python 路径、冻结 release 路径、私有配置路径、用户级最低权限、忽略重复实例、崩溃重试和无限执行时长。任务应由系统而非 SSH 会话托管。配置/任务脚本、备份及验收证据不进 Git；先前台验证三个实际分钟周期，再启动任务并核对定义与启动结果。未重启验证时不要宣称已验收开机自启。

## License

MIT。许可证仅适用于仓库代码和合成演示素材，不包含用户的私人数据。
