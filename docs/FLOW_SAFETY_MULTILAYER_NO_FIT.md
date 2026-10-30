# 多层内容安全增强版测试流程（无拟合版）

指标1.11不确定性量化建模、1.12不确定性来源分析和1.13多模态生成内容风险评估具体测试流程：

## 1 下载并上传交付包

在本机D盘新建MSRC_delivery文件夹，将交付网盘中的MSRC-code-netdisk-20261030.zip、SHA256SUMS.txt和prepared_data文件夹下载到其中。不下载fitted_only文件夹，保持目录和分卷原样，不手动解压或合并。

在电脑CMD执行，将USER、HOST替换为服务器用户名和地址；SSH端口不是22时同步替换：

```cmd
scp -P 22 -r "D:\MSRC_delivery" "USER@HOST:~/"
ssh -p 22 USER@HOST
```

## 2 核验材料并安装环境

```bash
cd ~/MSRC_delivery
sha256sum -c SHA256SUMS.txt
unzip MSRC-code-netdisk-20261030.zip
cd MSRC
bash scripts/01_setup_envs.sh --execute
```

校验均显示OK，安装成功提示：ENV_OK。后续服务器命令均在当前项目目录执行。

## 3 准备模型和数据

```bash
bash scripts/02_prepare_assets_data.sh --execute \
  --prepared-package ../prepared_data --download-models
bash scripts/02_prepare_assets_data.sh --check
```

脚本导入已处理数据和固定训练/测试划分，下载官方模型，不再重建数据集。成功提示：PREPARED_DATA_OK、FIXED_SPLITS_OK、ASSETS_OK、CONFIG_PATHS_OK。后续仅使用训练侧数据现场拟合。

## 4 采集信号、拟合并测试融合校准器

```bash
bash scripts/03_collect_factual.sh \
  --run-id ml_no_fit_001 --mode full --phase collect --execute
bash scripts/03_collect_factual.sh \
  --run-id ml_no_fit_001 --mode full --phase fit --execute
```

三个模型分别测试POPE全部9,000条和TextVQA验证集全部5,000条，仅在训练集拟合，固定参数后测试。成功提示：FACTUAL_COMPLETED ml_no_fit_001。

## 5 训练预警头并准备内容安全测试

```bash
bash scripts/04_score_safety.sh \
  --run-id ml_no_fit_001 --mode full --refit --execute
bash scripts/04_score_multilayer_variant.sh \
  --mode no_fit --run-id ml_no_fit_001 --phase prepare --execute
bash scripts/04_score_multilayer_variant.sh \
  --mode no_fit --run-id ml_no_fit_001 --phase check --execute
```

依次出现SAFETY_COMPLETED ml_no_fit_001、MULTILAYER_VARIANT_MATERIALS_OK no_fit后继续。

## 6 拟合并测试多层内容安全预警

```bash
bash scripts/04_score_multilayer_variant.sh \
  --mode no_fit --run-id ml_no_fit_001 --phase all --execute
```

脚本在训练侧拟合多层参数和确定阈值，固定后测试，测试集不参与拟合或选阈值。成功提示：MULTILAYER_VARIANT_SAFETY_COMPLETED ml_no_fit_001。

## 7 生成测试报告

```bash
bash scripts/05_report_multilayer_variant.sh \
  --mode no_fit --run-id ml_no_fit_001 --execute
cat runs/ml_no_fit_001/multilayer/joint/report.md
```

RUN_COMPLETED表示流程完成；ACCEPTANCE_PASS表示三个指标均达标，ACCEPTANCE_FAIL表示有指标未达标。未完成时显示INCOMPLETE。

三个指标报告均在runs/ml_no_fit_001/multilayer/joint/：指标1.11为signal_report.html；指标1.12为calibration_report.html，包含六组结果及平均提升率；指标1.13为warning_report.html。总汇总为report.md和report.html。

## 8 取回结果并打开报告

在服务器终端执行exit，回到本机CMD后执行：

```cmd
if not exist "D:\MSRC_results" mkdir "D:\MSRC_results"
scp -P 22 -r "USER@HOST:~/MSRC_delivery/MSRC/runs/ml_no_fit_001" "D:\MSRC_results"
start "" "D:\MSRC_results\ml_no_fit_001\multilayer\joint\report.html"
notepad "D:\MSRC_results\ml_no_fit_001\multilayer\joint\report.md"
```
