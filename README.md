# 人工智能课程作业

本目录包含奇异值分解、贝叶斯推断与交叉熵三部分作业，以及数值复核代码和配图。

主要提交文件：[作业_SVD_贝叶斯_交叉熵.md](作业_SVD_贝叶斯_交叉熵.md)。

## 文件说明

| 文件 | 内容 |
|---|---|
| `作业_SVD_贝叶斯_交叉熵.md` | 完整作业，含推导、数学建模、自设计例子与参考资料 |
| `verify_examples.py` | 复核三部分的数值与梯度，并重新生成配图 |
| `verification_results.json` | 数值复核结果，便于逐项检查 |
| `assets/*.svg` | 正文引用的矢量配图 |
| `assets/*.png` | 配图的位图版本 |
| `requirements.txt` | Python 依赖 |

贝叶斯案例采用“纳米光子学器件共振波长反演”，仿真工具为 Lumerical FDTD。全部光学数值为教学构造，尚未进行真实 FDTD 仿真或实验测量。原始课程课件为老师提供的学习资料。

## 数值复现

在本目录打开终端，执行：

```powershell
python -m pip install -r requirements.txt
python verify_examples.py
```

已安装 NumPy 与 Matplotlib 时，可直接执行第二行。脚本输出后验均值约 `7.940447 nm`、后验标准差约 `0.862796 nm`、平均交叉熵约 `0.363548 nat`。脚本同时核验 SVD 重构、贝叶斯数值积分与解析解、交叉熵梯度。

## 作业提交

仓库地址：[clt20031118-del/ai-course-homework](https://github.com/clt20031118-del/ai-course-homework)。
