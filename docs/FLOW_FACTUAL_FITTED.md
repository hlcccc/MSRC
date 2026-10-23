# 事实性版（有拟合版）具体测试流程

本版提供项目组已拟合的六个融合校准器及事实性预警阈值。验收方从公开来源准备基础模型和数据，加载所提供参数，重新生成回答并测试全部三个指标，不重新拟合。指标2.3评价回答错误风险，不评价内容有害性。

测试环境：已安装Git、NVIDIA驱动和Python 3.10（含venv）的Linux GPU服务器；参考环境为Ubuntu 22.04、16核CPU、约123 GiB内存、NVIDIA A100-SXM4-80GB和NVMe存储。默认GPU 0，须确认使用权限和空闲资源；整套公开材料建议预留不少于350 GiB磁盘。

依次执行以下六步，后续命令在项目根目录运行。独立运行编号为`factual_fitted_001`。

## （1）下载项目并安装环境

```bash
git clone --branch acceptance-four-variants-20261023 --single-branch \
  https://github.com/hlcccc/MSRC.git msrc_factual_fitted
cd msrc_factual_fitted
bash scripts/01_setup_envs.sh --execute
```

用途：获取固定代码版本，创建隔离环境并安装固定依赖。

成功信号：`ENV_OK`。

## （2）准备公开材料并下载已拟合参数

```bash
bash scripts/02_prepare_assets_data.sh \
  --execute --download-data --download-models --gpus 0
bash scripts/02_prepare_assets_data.sh --check --gpus 0
bash scripts/02_prepare_fitted.sh --execute
bash scripts/02_prepare_fitted.sh --check
```

用途：从公开来源下载固定基础权重和原始数据，重新处理并校验；从GitHub固定Release下载项目拟合参数，核验压缩包及内部文件SHA256。无需访问项目组服务器，不下载历史测试预测。公用材料与参数包也包含安全版本所需文件，但本版不执行内容安全模型或使用其分数。

测试基准：POPE三个子集各3,000条，共9,000条、500张原始图像；TextVQA官方验证集5,000条、3,166张原始图像。种子20260920，按图像身份分组对半划分，POPE训练/测试各4,500条，TextVQA训练2,504条、测试2,496条，同图不跨集。提供的校准器绑定该划分及对应模型，不得重新随机划分或挪用其他模型参数。

成功信号：`PUBLIC_INPUTS_MATCH`、`ASSETS_OK`、`CONFIG_PATHS_OK`和`FITTED_PARAMETERS_OK`，且校验命令正常结束。

参数位置：`assets/fitted/factual.json`。内容包括信号顺序、有效信号掩码、缺失填充值、标准化均值/标准差、融合系数/截距、训练身份及各格开发侧选定的预警阈值。

## （3）重新生成回答并采集信号

```bash
bash scripts/03_collect_factual.sh \
  --run-id factual_fitted_001 --gpus 0 \
  --mode full --phase collect \
  --parameters assets/fitted/factual.json --execute
```

用途：先验证并固定已拟合参数，再以LLaVA-1.5-13B、Qwen2.5-VL-7B、InternVL3-8B重新生成回答和信号。八类候选为改写一致性、重采样一致性、类型验证、跨模型一致性、依据核验、对象支持、序列置信度和输出熵。辅助回答只读，不改变基础模型。

当前统一采集器重新采集完整原始集合；训练侧新证据不用于更新参数或测试指标，最终只统计固定测试侧。

成功信号：`FACTUAL_COLLECTED factual_fitted_001`。

结果位置：`runs/factual_fitted_001/factual/evidence/`；本次冻结参数副本为`factual/provided_parameters.json`。

## （4）加载参数并测试指标2.1、2.2

```bash
bash scripts/03_collect_factual.sh \
  --run-id factual_fitted_001 --gpus 0 \
  --mode full --phase apply --execute
```

用途：原样加载第（3）步冻结的填充值、有效信号、标准化参数和融合系数，计算测试侧融合结果，不调用拟合。导出逐样本信号值、参与状态、系数和最终风险分。

指标2.1：同一次融合实际参与的信号不少于3种，不可用、训练侧恒定或零系数信号不计入。指标2.2：按15个等宽风险分箱，在同一测试集计算方法前与方法后ECE；方法前为1减去主回答平均token概率，方法后为融合校准风险概率。提升率＝（方法前ECE－方法后ECE）÷方法前ECE×100%，六组分别判定，均须≥10%。

成功信号：`FACTUAL_COMPLETED factual_fitted_001`。

结果位置：本运行目录下`factual/calibrators/`、`factual/signal_usage/`和`factual/report.json`。

## （5）加载冻结阈值并测试指标2.3

```bash
bash scripts/04_score_factual_warning.sh \
  --run-id factual_fitted_001 --execute
```

用途：使用参数包中各组合已冻结的事实性预警阈值，判定本次测试回答风险，不重新搜索阈值。POPE按回答是否错误标注；TextVQA按官方MMF/M4C软得分低于0.5标为错误。输出TP、TN、FP、FN，准确率＝（TP＋TN）÷测试样本数，六组分别判定是否≥75%。

预警在完整回答及信号提取后、受控本地输出前执行，不宣称逐token内容安全检测或线上流式接口已验证。

成功信号：`FACTUAL_WARNING_COMPLETED factual_fitted_001`。

结果位置：`runs/factual_fitted_001/factual_warning/`。

## （6）核验参数未变并汇总三个指标

```bash
bash scripts/05_score_report.sh \
  --run-id factual_fitted_001 --variant factual_fitted \
  --stage milestone --execute
```

用途：核验固定划分、六组数据覆盖和逐样本结果，确认使用参数与所提供文件一致、没有重新拟合，汇总三个指标。预期结果为各格实际信号≥3种、六格ECE提升均≥10%、六格事实性预警准确率均≥75%，实际结论以本次程序输出为准。

完成信号：`RUN_COMPLETED factual_fitted_001`仅表示完整运行，`ACCEPTANCE_PASS`才表示全部达标；`ACCEPTANCE_FAIL`表示有未达标项，`RUN_INCOMPLETE`表示核验未闭环。

测试报告：`runs/factual_fitted_001/report/report.html`；汇总数据：`runs/factual_fitted_001/summary.json`；参数来源记录：`runs/factual_fitted_001/VARIANT_VERIFIED.json`。

重复测试时统一更换第（3）至（6）步运行编号，仍使用同一公开参数包，但须重新推理和测试，不能将历史预测或历史报告作为本次结果。
