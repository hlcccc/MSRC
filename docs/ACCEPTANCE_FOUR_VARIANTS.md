# 四套测试流程

按本项目约定，“无拟合版”是不提供拟合参数、现场重新拟合；“有拟合版”是提供已拟合参数、只重新推理和测试。两种版本都不是直接复用历史测试分数。

| 版本 | 指标2.1、2.2 | 指标2.3 | 流程 |
|---|---|---|---|
| 事实性无拟合版 | 现场采集并拟合六个校准器 | 现场开发侧选阈值，测试回答错误风险 | [六步流程](FLOW_FACTUAL_NO_FIT.md) |
| 事实性有拟合版 | 新采集，加载六个校准器 | 加载开发侧冻结阈值，测试回答错误风险 | [六步流程](FLOW_FACTUAL_FITTED.md) |
| 内容安全无拟合版 | 现场采集并拟合六个校准器 | 现场训练PlugGuard头、选阈值，再测试 | [六步流程](FLOW_SAFETY_NO_FIT.md) |
| 内容安全有拟合版 | 新采集，加载六个校准器 | 加载PlugGuard头及冻结阈值，再测试 | [六步流程](FLOW_SAFETY_FITTED.md) |

每套均覆盖三个指标。六个事实性组合都使用当前八类候选信号和同一划分/推理预算，不拿历史十二信号结果冒充当前结果。

在固定版本项目根目录，也可用对应入口顺序完成同样流程：

```bash
bash scripts/run_factual_no_fit.sh --run-id factual_no_fit_001 --gpus 0 --execute
bash scripts/run_factual_fitted.sh --run-id factual_fitted_001 --gpus 0 --execute
bash scripts/run_safety_no_fit.sh --run-id safety_no_fit_001 --gpus 0 --execute
bash scripts/run_safety_fitted.sh --run-id safety_fitted_001 --gpus 0 --execute
```

一次选择一条，不要求四套连续运行；不加`--execute`只打印步骤。入口包含环境安装和材料准备，逐步流程与一键流程二选一，同一运行编号不能重复使用。

事实性指标三是回答错误风险，先完成生成及信号提取，再在受控本地输出前判定；内容安全指标三是PlugGuard逐生成token评分与门控。两个风险域不能互相替代验收结论。

## 参数交付

提供参数版本通过`scripts/02_prepare_fitted.sh`下载固定GitHub Release附件，指纹见`resources/reproduction/fitted.json`。附件只有项目拟合参数、划分身份和开发侧核验材料，不包含公开基础模型权重或历史测试预测。

六个校准器与原始全量事实性实验对应；PlugGuard头与修复跨ID重复请求泄漏后的split-v2全量实验对应。事实性预警阈值仅用该事实性实验开发侧选取，网格201点并列取最高；不为提高测试结果重新选参数。

## 验证边界

代码测试、参数导出/导入验证与新的GPU全流程验收是三件事。当前发布四套入口，不表示已重新跑完四套GPU全量实验。历史全量结果保留其原运行身份，不能替代新运行的输出。`RUN_COMPLETED`只表示完整运行，`ACCEPTANCE_PASS`才表示本次指标达标。

共享GPU规则仍适用：不占用其他用户全部显存、不干预其他进程；默认流程等待资源检查通过，不自动授权共享或降低核验标准。
