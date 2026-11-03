# 7. 测试流程

指标1.11不确定性量化建模、1.12不确定性来源分析和1.13多模态生成内容风险评估具体测试流程：

## 1 下载测试材料

代码及依赖补丁：本流程所在网盘目录MSRC_fast_single5090_20261103。

模型、数据及已有参数：https://pan.baidu.com/s/1P9IswryNxUndKQlqLV6M0Q　提取码：b3kw

下载两处的全部材料，合并放在服务器的~/MSRC_fast_single5090_20261103目录。三个模型文件夹与env、prepared_data、fitted_only并列放置，不另建models文件夹。

以下为直接运行流程。需要Slurm提交时，改按同目录的“单卡Slurm提交.md”执行。

## 2 核验材料并安装环境

依次运行下列命令。

```bash
cd ~/MSRC_fast_single5090_20261103
sha256sum -c SHA256SUMS.txt
cd env_patch
sha256sum -c SHA256SUMS.txt
cp wheels/*.whl ../env/wheels/
cd ..
unzip MSRC-fast-single5090-code-20261103.zip
cd MSRC
bash scripts/01_setup_envs.sh --execute \
  --offline-wheels ../env/wheels \
  --offline-python ../env/python/cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
```

文件校验均显示OK，环境安装成功提示：ENV_OK。后续命令均在当前项目目录执行。

## 3 准备模型、数据和校准模型

依次运行下列命令。

```bash
mkdir -p cache/fitted
cp ../fitted_only/parameters-20261023.tar.gz \
  cache/fitted/parameters.tar.gz
.venv/bin/python -B -m pipeline.provided_assets
.venv/bin/python -B -m pipeline.fast3_assets \
  --prepared-package ../prepared_data \
  --local-models ../local-models.fast3.json
```

成功提示：FITTED_PARAMETERS_OK、FIXED_SPLITS_OK、FAST3_ASSETS_OK。

## 4 运行指标测试（指标1.11、1.12、1.13）

运行下列命令，自动依次测试并合并结果。

```bash
bash scripts/run_fast_single.sh --run-id single_fitted_001
```

流程完成提示：RUN_COMPLETED。报告中SCOPED_PASS表示本次测试范围达标，SCOPED_FAIL表示未达标，INCOMPLETE表示未完成。

## 5 生成并查看测试报告

测试完成后自动生成报告，可运行下列命令查看。

```bash
cat runs/single_fitted_001/fast3/joint/report.md
```

总报告位置（项目根目录下，用浏览器打开）：

runs/single_fitted_001/fast3/joint/report.html
或
runs/single_fitted_001/fast3/joint/report.md

同目录下总报告及各指标报告：
总报告：report.html、report.md。
指标1.11 不确定性量化建模：signal_report.html、signal_report.md，查看不确定性信号数。
指标1.12 不确定性来源分析：calibration_report.html、calibration_report.md，查看校准性能提升。
指标1.13 多模态生成内容风险评估：warning_report.html、warning_report.md，查看高风险内容提前预警准确率。
