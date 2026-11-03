# 单卡Slurm提交

材料按测试流程第1步放好后，在下载目录执行：

```bash
cd ~/MSRC_fast_single5090_20261103
sha256sum -c SHA256SUMS.txt
cd env_patch
sha256sum -c SHA256SUMS.txt
cp wheels/*.whl ../env/wheels/
cd ..
unzip MSRC-fast-single5090-code-20261103.zip
cd MSRC
sbatch scripts/run_fast_single_slurm.sh --run-id single_fitted_001
```

如集群要求指定分区、账户或GPU型号，把管理员给的选项加在sbatch后、脚本路径前。脚本已申请单节点单卡；分区必须提供32GB级RTX 5090，且同一张卡不与其他计算任务共用。

提交脚本会在分配的计算节点内完成环境安装、材料准备和测试，无须在登录节点执行安装器的GPU检查。它使用Slurm分配的CUDA_VISIBLE_DEVICES，不要另传--gpu，也不要手动设置卡号。

查看队列与日志（将12345替换为sbatch返回的作业号）：

```bash
squeue -j 12345
tail -f slurm-12345.out
```

完成提示：RUN_COMPLETED。报告位置及判读同测试流程第5步。测试材料所在目录必须能被计算节点访问。

默认申请8核、48GB内存、24小时时限，管理员可通过sbatch选项调整。没有在你的Slurm集群实测过，不把提交成功等同于GPU测试通过。
