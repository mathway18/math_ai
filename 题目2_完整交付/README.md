# 2025 年高教社杯 B 题：问题二独立交付

本目录只处理问题二：附件 1（SiC，入射角 10°）和附件 2（SiC，入射角 15°）。
附件 Excel 是原始文件的只读副本；代码只在内存中剔除每个文件首个 0% 端点，不修改工作簿。

## 目录

```text
data/       附件1.xlsx、附件2.xlsx
code/       solve_q2.py、make_q2_figures.py
results/    审计、峰值初值、拟合参数、预测、厚度剖面
figures/    LaTeX 使用的 PDF 图和预览 PNG
paper/      问题二解题过程 LaTeX 及编译输出
```

## 复现

在本目录执行：

```powershell
C:\Python313\python.exe code\solve_q2.py
C:\Python313\python.exe code\make_q2_figures.py
```

需要 `numpy`、`scipy`、`openpyxl` 和 `matplotlib`。实际运行环境、输入文件 SHA-256、优化种子和全部关键数值会写入 `results/`。

## 本次运行结果

采用问题一的双光束 Fresnel 前向模型，并对两组角度共同反演厚度，得到

```text
d = 7.400452153 μm
附件1 R² = 0.999085523
附件2 R² = 0.998195359
1% MSE 局部敏感性区间 = [7.376503934, 7.422775503] μm
```

这里的 Lorentz--Drude 常数和每角度的线性仪器校准项属于明确记录的建模假设；它们不是题面附件直接提供的观测量，详见 `paper/q2_solution.tex`。

## 编译 LaTeX

```powershell
Set-Location paper
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build q2_solution.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build q2_solution.tex
```

最终 PDF 为 `paper/build/q2_solution.pdf`。
