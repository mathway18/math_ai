# 2025 B 题问题三独立交付包

本目录只对应题目三，未覆盖 `F:/数学建模/2025B` 下已经完成的问题一和问题二文件。
原始附件副本位于 `data/`，原始 Excel 不被求解程序修改。

## 主要结果

- 硅附件 3、4 的共同拟合波段：`2000--3400 cm^-1`。
- 硅主模型：部分多光束 Airy 诊断模型，`eta=0.711511282`。
- 硅共享厚度：`d=3.390020843 um`。
- 相对于双光束模型的 BIC 改善：`247.882121`。
- SiC 精确 Airy 修正厚度：`7.399475918 um`。
- 相对于问题二双光束参考值 `7.400452153 um` 的变化：`-0.000976235 um`，约 `-0.0132%`。

硅 Sellmeier 系数、Drude 代理参数和每个角度的有界幅值/基线校准是本实现
中为完成反演而明确记录的模型假设或 nuisance 参数，不应被解读为附件直接
测出的材料常数。硅主拟合的两个 Drude 代理参数触及搜索边界，因此结果文件
另附了衬底代理参数敏感性诊断。

## 目录说明

```text
code/solve_q3.py
    数据审计、硅双光束/完全相干/部分多光束拟合、SiC Airy 修正、敏感性分析
code/make_q3_figures.py
    只读取 results/ 中的结果和预测文件，生成论文图，不重新优化
data/附件1.xlsx ... 附件4.xlsx
    原始附件副本
data/B题.pdf
    题面副本
data/processed/
    仅在内存中删除首个精确 0% 端点后导出的审计 CSV
results/q3_result.json
    主结果、模型参数、指标、BIC、优化记录和输入审计摘要
results/q3_substrate_sensitivity.json
    硅衬底 Drude 代理参数条件敏感性
results/q3_si_predictions.csv
    硅选定波段的实测值和三种模型预测
results/q3_sic_correction_predictions.csv
    SiC 全谱 Airy 修正预测
results/data_audit.json
    四个附件的完整性、范围和 SHA-256 审计
figures/
    七幅论文图的 PDF、PNG 以及 figure_manifest.json
paper/q3_solution.tex
    详细解题过程 LaTeX 源文件
paper/build/q3_solution.pdf
    已编译论文
```

## 复现命令

在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'F:\数学建模\2025B\题目3_完整交付'
python code\solve_q3.py
python code\make_q3_figures.py
Set-Location -LiteralPath 'F:\数学建模\2025B\题目3_完整交付\paper'
xelatex -halt-on-error -output-directory=build `
  q3_solution.tex
xelatex -halt-on-error -output-directory=build `
  q3_solution.tex
```

求解脚本使用三个固定 Differential Evolution 种子 `2025`、`2718`、`3141`，
随后对完整数据使用 L-BFGS-B 精修；绘图脚本不会再次调用优化器。

