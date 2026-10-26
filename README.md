# Baby Crib Monitor

本地优先的婴儿床占用观察项目。包含**纯前端画面裁切工具**和**离线历史画面单图判断工具**。尚不包含实时采集服务。

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

需要 Node.js：

```sh
npm ci --include=dev
npx playwright install chromium
npm test
```

测试使用合成示意图，Chromium 沙箱开启。无需真实摄像头或任何账号。网站部署时只需复制 `tools/crop/` 中的两个静态文件；现有反代/登录保护由部署环境负责，不属于此工具。

## 离线历史数据运行器

`runtime/runner.py` 在运行机器上下载已选录像片段、定点抽帧、按导出的参数旋转裁切，然后向 OpenAI 兼容视觉接口发送**一张待判断图**。不使用参考图。前端工具仍无需后端。

```sh
python -m pip install -r runtime/requirements.txt
python -m unittest discover -s runtime -p 'test_*.py'
python runtime/runner.py /path/to/private/config.json
```

配置和数据必须放在仓库外。配置字段：`manifest`、`crop_config`（相对配置文件路径），`output_dir`、`model_base_url`、`model`、`interval_seconds`。采样清单由部署环境提前按固定间隔生成，每项包括 `id`、`epoch`、`timestamp`、`video_url`、`offset_seconds`；没有录像时 `video_url` 为 null，不拿别的时刻代替。录像与模型地址不写入源码，HTTPS 保持正常证书校验。

每帧输出原始 `yes / no / unclear` 及滤波结果。默认使用最近3次采样的多数决：至少2票才切换；`unclear` 不投票，平票或票数不足时保留上一状态；第一次明确结果用于初始化。缺失录像标记 `missing` 并重置滤波，不跨录像缺口延续状态。这是因果滤波，不使用未来帧、不回填转换时间。

输出包括 `frames/` 裁切图、可断点续跑的 `results.jsonl`、`progress.json`，完成后生成 `results.csv`、`summary.json`、`segments.json`。输出保留原始不明确/缺失标记，不能只看滤波结果推断这些时段都被成功观察。重跑必须使用相同冻结配置和采样清单。

尚未实现实时采集、自动发现录像或在线通知。本项目不做活体检测，也不把床位占用信号解释为已确认睡眠。

## License

MIT。许可证仅适用于仓库代码和合成演示素材，不包含用户的私人数据。
