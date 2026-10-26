# 许可证边界（不是法律意见）

## 已核对的上游依据

- 固定版本完整 LICENSE：<https://github.com/ultralytics/ultralytics/blob/v8.4.189/LICENSE>。核读内容为 GNU Affero General Public License v3，不能写成 MIT。
- 官方许可说明：<https://www.ultralytics.com/license>。官方 FAQ 明确将 code、models、architectures、training pipelines、trained/fine-tuned models 纳入其 AGPL-3.0 / Enterprise 许可说明；对相应整体派生项目，说明中还提及在适用情况下的 model weights。
- 模型官方说明：<https://docs.ultralytics.com/models/yolo11/>、<https://docs.ultralytics.com/models/yolo26/>。

以上是上游许可文本与权利人主张的记录，不等同于对本项目组合方式、用户私人使用或具体法律义务的确定裁定。动态网页可能变化，正式发布审核时应再次核对。

## 不可混淆的四层

| 范围 | 项目处理 |
| --- | --- |
| 自有通用 glue、文档、合成/mock 测试 | 项目 MIT；不宣称集成后全栈 MIT |
| Ultralytics 实现 | 独立 AGPL-3.0 边界；不 vendor，不复制实现进入自有 MIT 文件 |
| 官方预训练及其微调/派生 checkpoint | 按上游 AGPL-3.0 / Enterprise 边界审查，不冒称 MIT，不随仓库分发 |
| 用户真实图像、标注、运行输出和配置 | 不是开源演示资产，没有授权公开，永远留在 Git 外 |

不带 vendor、不分发 `.pt`、进程分离、写一个 MIT LICENSE 或称作“私有运行”，**都不能自动证明所有适用义务已经满足**。对调用 AGPL 实现的 glue，组合、分发、修改和网络交互等事实仍可能影响对应源码提供和兼容许可要求；可选安装不是当然豁免。

## 公开前的阻断项

1. 确认要发布的源码范围及版权来源；不要把整个训练集成包宣传成“MIT Ultralytics 框架”。
2. 对实际部署/分发方式和微调权重义务，取得合适的专业意见或 Ultralytics 明确答复；需要时评估 Enterprise 或替代技术。**本项目不替用户决定已满足 AGPL，也不假设已有 Enterprise。**
3. 用户明确禁止公开训练后权重、家庭图像与敏感资料。若许可审查发现冲突，停止相关发布并上报，不能以“开源义务”为由擅自公开用户资料或给权重改许可证。
4. 发布时保留适用 notices/依赖许可信息，核对全部依赖；Torch/torchvision 等各有其许可，不能统称 MIT。

本文件不提供绕过许可证的方法，也不承诺只公开 glue 就足以满足任何组合项目的全部义务。
