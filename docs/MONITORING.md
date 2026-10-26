# 通用 Grafana 面板设计

此处只给查询契约，不导出任何现有 dashboard；没有真实 UID、datasource、主机或地址。与 `runtime/metrics.py` 对照使用，接入后仍须核对目标机实际输出。

在 Grafana 手动选择自己的 Prometheus datasource，创建 `job` 和 `instance` 变量。查询均显式选择 `{job="$job",instance="$instance"}`，避免聚合不同摄像头/进程。不要在公开模板中填写真实变量默认值。状态映射：1=目标内有人、0=无、-1=未知/缺失；不能映射成睡眠/清醒。

| 面板 | PromQL（以下 S 表示相同 job/instance selector，不是实际 PromQL 变量） |
| --- | --- |
| 原始当前值 / 时间线 | `crib_monitor_state{S,kind="raw"}` |
| 因果滤波保持值 / 时间线 | `crib_monitor_state{S,kind="filtered"}` |
| Prometheus scrape 可达 | `up{S}` |
| worker 主线程新鲜 | `crib_monitor_up{S}` |
| 采样年龄（秒） | `time() - crib_monitor_last_sample_timestamp_seconds{S}` |
| 技术成功年龄（秒） | `time() - crib_monitor_last_success_timestamp_seconds{S}` |
| 当前 ROI 可用 | `crib_monitor_roi_valid{S}` |
| 当前目标 confidence | `crib_monitor_yolo_confidence{S}` |
| 最近各阶段秒数 | `crib_monitor_stage_duration_seconds{S}`，legend=`{{stage}}` |
| 最近结果增量 | `increase(crib_monitor_samples_total{S}[$__rate_interval])`，legend=`{{result}}` |
| worker 启动时间 | `crib_monitor_process_start_time_seconds{S}` |

例如 raw 面板的完整语法是：

```promql
crib_monitor_state{job="$job",instance="$instance",kind="raw"}
```

## 通用接入模板

由所有者复制到私有监控配置后填写自己的目标；以下 `example.invalid` 不可用，不是任何实际机器地址。不要把真实 target、认证信息或数据源 UID 回填到本仓库。

```yaml
scrape_configs:
  - job_name: crib-monitor
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 5s
    static_configs:
      - targets: ["metrics.example.invalid:9000"]
```

exporter 的监听端口/接口须与私有 target 匹配，并在部署侧同时配置网络防火墙与 HTTP 客户端白名单。Grafana 手动选已有 Prometheus 数据源后，按上述查询创建 Stat、State timeline 和 Time series 面板；无需模型响应或图片。只添加本项目 job/面板，不覆盖无关 dashboard。datasource/instance 的实际值与 dashboard 导出文件均留私有。

## 必须保留的语义

- `up` 是 scrape；`crib_monitor_up` 是 worker 心跳新鲜；二者都不等于观察成功，更不等于安全。
- filtered 可能保持旧值；raw=unclear/missing、无 ROI、年龄过大或 scrape 无数据时，不能继续显示成可靠的当前占用。面板必须显式显示 unknown/no data，不跨缺口连接/填充历史状态。
- last success 为完整有效模型响应完成时间，`unclear` 也可算**技术成功**，不能当作明确 yes/no 的新鲜度。
- timestamp 初始化 0 时显示“尚无样本/尚无技术成功”，不要把超大年龄解释为长期运行；no data 与 0 也要区分。
- counter 每个采样只进一个桶，有技术错误时 error 优先；重启归零。`increase` 有外推语义，不能当作精确历史事件个数，精确审计用私有 JSONL。
- 当前 duration 是 gauge，不是 histogram。不能对该指标计算全历史 P95 或 `histogram_quantile`；未执行阶段为0，失败阶段已耗时亦保留。capture 含取流/关键帧等待、原图解码/hash；detect 含解码、推理、NMS、裁切/缩放/JPEG写入及复读/hash；qwen 含编码、请求和解析，无框跳过为0。total 含观察/滤波，不含结果/心跳落盘或调度等待；初始导入/权重加载不计，首次 lazy backend warmup 计入 detect。
- ROI/confidence 不检测孩子、遮挡或画外状态。任何告警门槛必须按真实 timeout/间隔验证，不把 dashboard 变成婴儿安全系统。

## 接入验收

核对 metric 名、kind/result/stage 枚举、秒单位、默认90秒心跳门槛（运行配置可改）、初始化与重启/缺口语义；实测 no frame/no box/Qwen timeout、scrape stale、进程退出、PM2 重启后的面板。只读验证 `/metrics` 精确路由及其它路由拒绝，确认指标/labels/HELP 不泄露地址、ID、prompt 或图像。不得将真实 dashboard JSON 或截图直接当通用模板提交。
