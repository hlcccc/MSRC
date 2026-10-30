# 网盘交付维护

本页供打包维护使用，不增加验收人员的测试步骤。

代码ZIP按发布提交生成；两份Word、Markdown及ZIP内流程保持一致。代码ZIP更新后，在上传前重新生成校验表：

```bash
python scripts/build_netdisk_checksums.py /path/to/MSRC_fitted /path/to/MSRC_no_fit
```

脚本重新计算实际文件的SHA256，固定输出UTF-8无BOM、LF换行。不要使用Windows文本编辑器另存校验表。Linux上再分别执行两个目录及fitted_only目录中的`sha256sum -c SHA256SUMS.txt`，全部显示OK后上传。

只修改代码或流程时，替换两个目录的代码ZIP和根校验表；修复换行时还需替换fitted_only内的校验表。数据分卷及参数压缩包未变化时不重传。
