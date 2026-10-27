# 人工智能课程作业

本仓库收录人工智能课程作业，以及相应的数值复核代码与资料来源。

## 作业目录

| 作业 | 主要提交文件 | 内容 |
|---|---|---|
| SVD、贝叶斯与交叉熵 | [作业_SVD_贝叶斯_交叉熵.md](作业_SVD_贝叶斯_交叉熵.md) | 推导、建模案例与数值复核 |
| ResNet | [作业_ResNet_梯度推导与网络对比.md](作业_ResNet_梯度推导与网络对比.md) | 权重梯度推导、Highway 与 ResNet 对比、DenseNet 优缺点 |

## ResNet 作业复核

正文包含完整推导与原始论文引用。补充脚本仅依赖 Python 3 标准库：

```powershell
python verify_resnet_gradients.py
```

脚本核验两个非线性残差单元的权重、偏置与输入梯度，以及 Highway 门控雅可比，结果保存为 `verification_resnet_results.json`。全部数值为教学构造，没有进行图像数据集训练。

以下文件说明与原有数值复现命令对应 SVD、贝叶斯与交叉熵作业。

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

