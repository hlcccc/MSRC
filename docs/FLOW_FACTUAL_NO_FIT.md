# 事实性版（无拟合版）具体测试流程

本版不附带已拟合参数。验收方从公开基础模型和原始数据开始，按流程重新拟合，再测试指标2.1、2.2和2.3。“无拟合版”表示未提供拟合参数，不表示测试过程中禁止拟合。指标2.3评价回答错误风险，不评价内容有害性。

测试环境：已安装Git、NVIDIA驱动和Python 3.10（含venv）的Linux GPU服务器；参考环境为Ubuntu 22.04、16核CPU、约123 GiB内存、NVIDIA A100-SXM4-80GB和NVMe存储。默认GPU 0，须事先确认有使用权限和足够空闲资源。整套公开材料建议预留不少于350 GiB磁盘。

在同一终端依次执行六步，进入项目后均在项目根目录运行。以下使用独立运行编号`factual_no_fit_001`。

## （1）下载项目并安装环境

```bash
git clone --branch acceptance-four-variants-20261023 --single-branch \
  https://github.com/hlcccc/MSRC.git msrc_factual_no_fit
cd msrc_factual_no_fit
bash scripts/01_setup_envs.sh --execute
```

用途：获取本流程对应的固定代码版本，创建隔离环境并安装固定依赖。

成功信号：`ENV_OK`。

## （2）下载模型和数据并重新处理

```bash
bash scripts/02_prepare_assets_data.sh \
  --execute --download-data --download-models --gpus 0
bash scripts/02_prepare_assets_data.sh --check --gpus 0
```

用途：从固定公开来源下载基础权重和原始材料，重新处理数据并校验。该公用准备脚本也准备安全基准和良性样本，但本版不运行内容安全测试，不将其计入事实性指标。

测试基准：POPE的random、popular、adversarial各3,000条，共9,000条、500张原始图像；TextVQA官方验证集5,000条、3,166张原始图像。种子20260920，按图像身份分组对半划分；POPE训练4,500条、测试4,500条，TextVQA训练2,504条、测试2,496条。同图不跨集，1:1为图像组比例。

成功信号：`PUBLIC_INPUTS_MATCH`、`ASSETS_OK`、`CONFIG_PATHS_OK`，且最后的校验命令正常结束。

## （3）重新生成回答并采集信号

```bash
bash scripts/03_collect_factual.sh \
  --run-id factual_no_fit_001 --gpus 0 \
  --mode full --phase collect --execute
```

用途：LLaVA-1.5-13B、Qwen2.5-VL-7B和InternVL3-8B分别在两套数据上重新生成回答，采集八类候选信号：改写一致性、重采样一致性、类型验证、跨模型一致性、依据核验、对象支持、序列置信度和输出熵。跨模型一致性只读取同题回答，不更新基础模型。

成功信号：`FACTUAL_COLLECTED factual_no_fit_001`。

结果位置：`runs/factual_no_fit_001/factual/inputs/`、`evidence/`和`protocol.json`。

## （4）现场拟合并测试指标2.1、2.2

```bash
bash scripts/03_collect_factual.sh \
  --run-id factual_no_fit_001 --gpus 0 \
  --mode full --phase fit --execute
```

用途：仅在本次训练侧确定缺失填充值、有效信号、标准化参数和融合系数，分别拟合六个融合校准器；再在训练侧选择事实性预警阈值。阈值网格为0至1共201点，训练准确率并列时取最高阈值。固定参数后测试，不依据测试标签调参。

指标2.1：导出逐样本信号值、参与状态、融合系数和最终风险分；同一次融合中实际参与的信号不少于3种，不可用、训练侧恒定或零系数的信号不计入。

指标2.2：在同一测试集按15个等宽风险分箱计算ECE。方法前为1减去主回答平均token概率，方法后为融合校准风险概率；提升率＝（方法前ECE－方法后ECE）÷方法前ECE×100%。六个组合分别判定，均须≥10%。

成功信号：`FACTUAL_COMPLETED factual_no_fit_001`。

结果位置：`factual/calibrators/`、`factual/signal_usage/`和`factual/report.json`，均位于本运行目录内。

## （5）测试指标2.3事实性风险预警

```bash
bash scripts/04_score_factual_warning.sh \
  --run-id factual_no_fit_001 --execute
```

用途：对本次测试侧回答使用第（4）步冻结的校准器和阈值判定风险，并在受控本地输出前完成预警。POPE按目标存在性回答是否错误标注；TextVQA按官方MMF/M4C软得分低于0.5标为错误。准确率＝（TP＋TN）÷测试样本数，各组合分别输出混淆矩阵并判定是否≥75%。

本步骤在完整回答和信号提取后、受控本地输出前执行，不代表已验证逐token内容安全检测或在线流式接口。

成功信号：`FACTUAL_WARNING_COMPLETED factual_no_fit_001`。

结果位置：`runs/factual_no_fit_001/factual_warning/`，包含逐样本判定、受控输出记录和`report.json`。

## （6）核验版本并汇总三个指标

```bash
bash scripts/05_score_report.sh \
  --run-id factual_no_fit_001 --variant factual_no_fit \
  --stage milestone --execute
```

用途：核验数据覆盖、训练/测试隔离、现场拟合口径及逐样本结果，汇总三个指标。预期结果为指标2.1各组合实际参与信号≥3种、指标2.2六组ECE提升均≥10%、指标2.3六组事实性预警准确率均≥75%，以本次程序输出判定。

完成信号：`RUN_COMPLETED factual_no_fit_001`只表示测试完整、报告生成；`ACCEPTANCE_PASS`才表示三项指标均达标。`ACCEPTANCE_FAIL`表示有指标未达标，`RUN_INCOMPLETE`表示测试或核验不完整。

测试报告：`runs/factual_no_fit_001/report/report.html`；汇总数据：`runs/factual_no_fit_001/summary.json`；版本来源记录：`runs/factual_no_fit_001/VARIANT_VERIFIED.json`。

重复测试时统一更换第（3）至（6）步的运行编号，重新采集、拟合和测试，不覆盖旧结果。不加载项目组已拟合参数或历史测试预测；公开基础权重不重新预训练。
