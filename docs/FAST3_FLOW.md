# 三张5090测试流程（简明版）

使用三台独占RTX 5090服务器，指标一、二只测InternVL，指标三不变。加载已有参数，不进行现场训练。

## 1 下载材料

代码和流程：https://pan.baidu.com/s/1ycuHzMj57an0HlKoJBCW0Q　提取码：uxg2

模型、数据和环境：https://pan.baidu.com/s/1P9IswryNxUndKQlqLV6M0Q　提取码：b3kw

两个分享全部下载，合并到 D:\MSRC_fast_3x5090_20261103。三个模型文件夹及env、prepared_data、fitted_only直接放在该目录下，不另建models文件夹。

## 2 上传到三台服务器

第2至4步在三台各做一次。三台使用相同用户名和安装路径；每台建议Ubuntu 22.04、16核、60GB内存、180GB空闲盘。

在电脑CMD执行，USER、HOST、PORT换成当前服务器的用户名、地址、SSH端口：

```cmd
scp -P PORT -r "D:\MSRC_fast_3x5090_20261103" "USER@HOST:~/"
ssh -p PORT USER@HOST
```

## 3 安装环境

登录服务器后执行：

```bash
cd ~/MSRC_fast_3x5090_20261103
sha256sum -c SHA256SUMS.txt
cd env_patch
sha256sum -c SHA256SUMS.txt
cp wheels/*.whl ../env/wheels/
cd ..
unzip MSRC-fast3-code-20261103.zip
cd MSRC
bash scripts/01_setup_envs.sh --execute \
  --offline-wheels ../env/wheels \
  --offline-python ../env/python/cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
.venv/bin/python -c 'import torch; assert torch.__version__ == "2.7.1+cu128"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

校验全部显示OK，安装提示ENV_OK，最后显示RTX 5090。后续服务器命令均在当前项目目录执行。

## 4 准备数据和参数

```bash
mkdir -p cache/fitted
cp ../fitted_only/parameters-20261023.tar.gz cache/fitted/parameters.tar.gz
.venv/bin/python -B -m pipeline.provided_assets
.venv/bin/python -B -m pipeline.fast3_assets \
  --prepared-package ../prepared_data \
  --local-models ../local-models.fast3.json
```

成功提示：FITTED_PARAMETERS_OK、FIXED_SPLITS_OK、FAST3_ASSETS_OK。

## 5 设置三台连接

从本步起，只在第一台操作。编辑配置中的host和port，分别填写第二、三台的USER@HOST及端口；第一台须能免密SSH到另外两台。

```bash
cp configs/fast3_hosts.example.json configs/fast3_hosts.json
nano configs/fast3_hosts.json
ssh -o BatchMode=yes -p PORT1 USER1@HOST1 'hostname'
ssh -o BatchMode=yes -p PORT2 USER2@HOST2 'hostname'
nvidia-smi
```

两条SSH命令均显示对应主机名后继续；三张GPU应空闲。

## 6 启动测试

建议在tmux中执行以下命令，避免断线中断：

```bash
.venv/bin/python -B -u -m pipeline.fast3 \
  --phase cluster --run-id fast3_fitted_001 \
  --hosts configs/fast3_hosts.json --gpu 0
```

程序自动分配三卡任务，完成生成、审核、合并和报告，不需要分别启动。仅测试固定测试集：POPE 4500条、TextVQA 2496条；安全3360请求、五个实验臂不变。

## 7 查看结果

```bash
cat runs/fast3_fitted_001/fast3/joint/report.md
cat runs/fast3_fitted_001/fast3/joint/summary.json
cat runs/fast3_fitted_001/fast3/cluster_timing.json
```

RUN_COMPLETED表示流程完成；SCOPED_PASS表示本次范围达标；SCOPED_FAIL表示未达标；INCOMPLETE表示未完成。

三个指标报告为signal_report、calibration_report、warning_report，均在fast3/joint目录；总耗时见cluster_timing.json。

## 8 取回结果并打开报告

服务器执行exit，回到电脑CMD。以下PORT0、USER0、HOST0填第一台的连接信息：

```cmd
if not exist "D:\MSRC_results" mkdir "D:\MSRC_results"
scp -P PORT0 -r "USER0@HOST0:~/MSRC_fast_3x5090_20261103/MSRC/runs/fast3_fitted_001" "D:\MSRC_results"
start "" "D:\MSRC_results\fast3_fitted_001\fast3\joint\report.html"
notepad "D:\MSRC_results\fast3_fitted_001\fast3\joint\report.md"
```

中断续跑可再次执行第6步同一命令，不要另开同编号任务。本版尚待三卡全量验收，是否达标以本次报告为准；技术细节和审计要求另见逐步审计.md。
