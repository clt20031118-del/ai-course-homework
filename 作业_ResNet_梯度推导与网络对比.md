# ResNet 课后作业：梯度推导与网络对比

作答日期：2026 年 10 月 27 日

## 题目

根据课堂照片，本次作业包含以下三问：

1. 利用 $x_{l+1}=x_l+F(x_l,W_l)$，推导损失对权重 $W_l$ 的梯度，并说明它与深层梯度 $\partial\mathcal L/\partial x_{l+1}$ 的关系。
2. 查阅资料，比较 Highway Network 与 ResNet 在快捷连接实现上的异同。
3. 说明 DenseNet 与 ResNet 连接方式的本质区别，并分析 DenseNet 的优缺点。

---

## 1. 理论推导：ResNet 的权重梯度

### 1.1 符号与前提

记最终损失为标量 $\mathcal L$，第 $l$ 个残差单元的输入为 $x_l$，残差函数为 $F(x_l,W_l)$。按题目给定的模型：

$$
x_{l+1}=x_l+F(x_l,W_l).
$$

这里假设快捷分支是恒等映射，且相加后没有额外的非线性操作。$W_l$ 表示该单元中待求导的参数；若它只代表单元内部某一层的权重，对 $F$ 求导时还要包含该层之后的运算。各单元的参数默认不共享。

以下将特征展平为列向量，并采用**列梯度**约定，以明确矩阵乘法的顺序。上述恒等条件及其传播分析可参见 [He 等，2016，§2](https://arxiv.org/html/1603.05027v3#S2)。

### 1.2 对权重应用链式法则

先看某个权重分量 $w_{l,j}$。由于 $x_l$ 由前面单元决定，与本单元未共享的权重无关，因此：

$$
\frac{\partial x_l}{\partial w_{l,j}}=0,
\qquad
\frac{\partial x_{l+1,i}}{\partial w_{l,j}}
=
\frac{\partial F_i(x_l,W_l)}{\partial w_{l,j}}.
$$

应用链式法则：

$$
\boxed{
\frac{\partial\mathcal L}{\partial w_{l,j}}
=
\sum_i
\frac{\partial\mathcal L}{\partial x_{l+1,i}}
\frac{\partial F_i(x_l,W_l)}{\partial w_{l,j}}
}
$$

将权重展平为 $w_l=\operatorname{vec}(W_l)$，定义：

$$
g_{l+1}=\nabla_{x_{l+1}}\mathcal L,\qquad
J_{W_l}=\frac{\partial F(x_l,W_l)}{\partial w_l}.
$$

其中 $g_{l+1}\in\mathbb R^n$，$J_{W_l}\in\mathbb R^{n\times p}$，$p$ 为参数个数。矩阵形式为：

$$
\boxed{
\nabla_{w_l}\mathcal L
=
J_{W_l}^{\mathsf T}g_{l+1}
}
$$

**它与深层梯度的关系是：深层传回的 $g_{l+1}$ 作为上游信号，经过本残差分支对权重的局部雅可比，形成权重梯度。** 恒等分支不含 $W_l$，因此权重梯度中没有额外的“$+1$”；“$+I$”出现在对输入特征求导时。

若损失另含正则项 $\lambda R(W_l)$，还应加上 $\lambda\nabla_{w_l}R$。上式对应题目中的数据损失。

### 1.3 对输入求导：恒等分支的作用

定义：

$$
J_l=\frac{\partial F(x_l,W_l)}{\partial x_l}.
$$

则单元的输入输出雅可比为：

$$
\frac{\partial x_{l+1}}{\partial x_l}=I+J_l.
$$

由链式法则：

$$
\boxed{
g_l=(I+J_l)^{\mathsf T}g_{l+1}
=g_{l+1}+J_l^{\mathsf T}g_{l+1}
}
$$

第一项 $g_{l+1}$ 沿恒等分支直接传回，第二项通过残差分支传回。若没有快捷连接，对应公式只有 $J_l^{\mathsf T}g_{l+1}$。

对于更深的特征 $x_L$，连续使用这一递推可得：

$$
g_{l+1}
=
(I+J_{l+1})^{\mathsf T}
(I+J_{l+2})^{\mathsf T}
\cdots
(I+J_{L-1})^{\mathsf T}g_L.
$$

因此，本单元的权重梯度也可写成：

$$
\nabla_{w_l}\mathcal L
=
J_{W_l}^{\mathsf T}
(I+J_{l+1})^{\mathsf T}\cdots
(I+J_{L-1})^{\mathsf T}g_L.
$$

这说明后续各层通过 $g_{l+1}$ 影响当前权重的更新，当前残差函数的局部导数则决定这些信号如何作用于每个参数。

也可以由前向递推得到：

$$
x_L=x_l+\sum_{i=l}^{L-1}F(x_i,W_i),
$$

从而：

$$
g_l=
\left[
I+
\frac{\partial}{\partial x_l}
\sum_{i=l}^{L-1}F(x_i,W_i)
\right]^{\mathsf T}g_L.
$$

这里求导必须计入后续 $x_i$ 对 $x_l$ 的依赖。该加法分解与论文中的传播分析一致：[Identity Mappings in Deep Residual Networks，式 (4)–(5)](https://arxiv.org/html/1603.05027v3#S2)。

### 1.4 一个具体的权重层例子

设简化残差分支为：

$$
F(x_l,W_l)=\phi(W_lx_l+b_l),
\qquad
z_l=W_lx_l+b_l,
$$

其中 $\phi$ 逐元素作用。令：

$$
\delta_l=g_{l+1}\odot\phi'(z_l).
$$

则：

$$
\boxed{
\nabla_{W_l}\mathcal L=\delta_lx_l^{\mathsf T},
\qquad
\nabla_{b_l}\mathcal L=\delta_l,
\qquad
g_l=g_{l+1}+W_l^{\mathsf T}\delta_l
}
$$

$\odot$ 表示逐元素乘法。若 $\phi$ 为恒等映射，则 $\nabla_{W_l}\mathcal L=g_{l+1}x_l^{\mathsf T}$。实际多层卷积残差分支仍适用链式法则，只需继续在分支内部反向传播。

### 1.5 适用范围与解释

- **恒等路径提供直接传播项，但不保证总梯度永不消失。** 例如人为令 $F(x)=-x$，则 $I+J_l=0$，总梯度仍为零；残差分支对权重的导数也可能为零。因此应表述为“改善梯度传播、缓解深层优化困难”。
- **原始 ResNet 的相加后 ReLU 需要另求导。** 若 $y_l=x_l+F(x_l,W_l)$、$x_{l+1}=\rho(y_l)$，记 $D_l=\operatorname{diag}(\rho'(y_l))$，则 $g_l=(I+J_l)^{\mathsf T}D_l^{\mathsf T}g_{l+1}$，而权重梯度为 $J_{W_l}^{\mathsf T}D_l^{\mathsf T}g_{l+1}$。
- **投影快捷连接需要替换恒等项。** 若 $x_{l+1}=P_lx_l+F(x_l,W_l)$，则 $g_l=(P_l+J_l)^{\mathsf T}g_{l+1}$。若 $P_l$ 也可训练，其参数另有梯度。

相加后的激活与维度变化会影响恒等传播条件，论文对此作了区分：[原始 ResNet，§3.2](https://arxiv.org/html/1512.03385v1#S3.SS2)、[Identity Mappings，§2](https://arxiv.org/html/1603.05027v3#S2)。

---

## 2. 思考与对比（一）：Highway Network 与 ResNet

### 2.1 两种快捷连接的表达式

Highway Network 在变换分支和传递分支上设置门控：

$$
y=H(x,W_H)\odot T(x,W_T)+x\odot C(x,W_C).
$$

$T$ 为变换门，$C$ 为传递门。原论文采用简化设置 $C=1-T$：

$$
\boxed{
y=H(x)\odot T(x)+x\odot[1-T(x)]
}
$$

通常 $T(x)=\sigma(W_Tx+b_T)$。门值由输入与可训练参数决定；$T$ 接近零时更偏向保留输入，接近一时更偏向非线性变换。原论文还使用负的门偏置，使初始状态更偏向传递输入。[Highway Networks，§2](https://arxiv.org/html/1505.00387v2#S2)

ResNet 的恒等快捷连接则为：

$$
\boxed{
y=x+F(x,W)
}
$$

两分支相加时系数固定为一，恒等快捷分支本身没有可训练门控参数；维度不匹配时可以使用投影。[Deep Residual Learning for Image Recognition，§2–3](https://arxiv.org/html/1512.03385v1#S3)

### 2.2 异同比较

| 比较维度 | Highway Network | ResNet |
|---|---|---|
| 共同思路 | 通过跨层连接帮助信息与梯度传播 | 同样通过跨层连接改善深层网络优化 |
| 两分支如何融合 | 门控后的逐元素加法 | 输入与残差逐元素相加 |
| 快捷分支系数 | $C(x)$，常取 $1-T(x)$，随输入变化 | 恒等快捷连接中固定为 $1$ |
| 额外参数 | 门控网络引入参数与计算 | 恒等快捷连接没有参数；投影连接有参数 |
| 如何接近恒等映射 | 令 $T(x)$ 接近 $0$ | 令残差 $F(x)$ 接近 $0$ |
| 主要设计特点 | 自适应调节变换与保留的比例 | 围绕输入学习增量修正，结构简洁 |

从梯度上也能看到区别。对耦合门 Highway 层，记 $J_H=\partial H/\partial x$、$J_T=\partial T/\partial x$，则直接求导得：

$$
\frac{\partial y}{\partial x}
=
\operatorname{diag}(1-T)
+\operatorname{diag}(T)J_H
+\operatorname{diag}(H-x)J_T.
$$

这里第三项来自门值对输入的依赖，不能将门值当作常数而省略。相比之下，题目所用 ResNet 模型的雅可比直接为 $I+J_F$。

**结论：两者都建立跨层通路；Highway 使用可学习的门控制信息流，ResNet 使用恒等通路叠加残差。** 还需注意：在 $C=1-T$ 的耦合门形式下，直接令 $T=1$ 会得到 $y=H(x)$，不能直接得到 $y=x+F(x)$。若从独立的双门形式类比，需将两个系数都固定为一，并将变换分支定义为 $F$。

---

## 3. 思考与对比（二）：DenseNet 与 ResNet

### 3.1 本质区别：相加与拼接

ResNet 在相应单元中将输入与残差**逐元素相加**：

$$
x_l=x_{l-1}+F_l(x_{l-1}).
$$

DenseNet 在同一个 dense block 内，把此前所有层的特征**沿通道维拼接**，作为当前层的输入：

$$
\boxed{
x_l=H_l([x_0,x_1,\ldots,x_{l-1}])
}
$$

其中 $[\cdot]$ 表示通道拼接；$x_l$ 只表示当前层新产生的特征。拼接要求空间尺寸一致，因此跨尺度时通过 transition layer 连接不同 dense block。[Densely Connected Convolutional Networks，§3](https://arxiv.org/html/1608.06993v5#S3)

| 比较维度 | ResNet | DenseNet |
|---|---|---|
| 融合操作 | 逐元素相加 | 沿通道维拼接 |
| 历史特征的保留 | 融入更新后的特征状态 | 在 block 内作为独立通道保留 |
| 当前变换的输入 | 标准单元的当前特征状态 | 同一 block 中此前所有层的特征 |
| 尺寸要求 | 相加两端的形状必须一致，必要时投影 | 拼接端的空间尺寸一致，通道数可以不同 |
| 特征宽度 | 同一 stage 通常保持固定宽度 | 累积特征通道数随层数增长 |

例如输入有 $k_0=64$ 个通道，每层产生 $k=32$ 个新通道，则第 1、2、3 层的输入分别为 $64$、$96$、$128$ 个通道，三层后累积表示有 $160$ 个通道。一般地：

$$
C_l^{\mathrm{in}}=k_0+(l-1)k,\qquad
C_{\mathrm{block}}^{\mathrm{out}}=k_0+mk,
$$

其中 $m$ 为 block 中的层数，$k$ 称为增长率。该例由上述通道递推直接计算。

### 3.2 优点

1. **复用特征。** 后续层可以直接使用早期特征，每层只需补充少量新特征。
2. **改善梯度传播。** 密集连接带来多条较短的反向传播路径，有助于训练深层网络。
3. **提高参数利用效率。** 结合较小增长率、瓶颈层和过渡层压缩，可以用较紧凑的模型达到较高精度；这不意味着任意 DenseNet 都比任意 ResNet 参数更少。

这些设计与实验结论见 [DenseNet 原论文，§3、§5](https://arxiv.org/html/1608.06993v5#S5)。

### 3.3 缺点与代价

1. **训练显存压力可能较大。** 虽然参数可以较少，训练仍要保存激活；朴素实现还可能重复保存拼接和归一化结果，使相关存储随深度呈二次增长。
2. **后续层的输入不断变宽。** 拼接、数据搬运与特征读取会增加实现成本，实际速度取决于实现与硬件，不能只由参数量判断。
3. **需要控制通道增长。** 可用 $1\times1$ 瓶颈卷积与过渡层压缩；也可用共享存储、重计算等方式减轻显存开销。

其中，朴素实现的二次激活存储开销具有实现依赖性，并非任何 DenseNet 实现都必须如此；内存优化方案可以改善它。[Memory-Efficient Implementation of DenseNets](https://arxiv.org/abs/1707.06990)

**因此，ResNet 以相加方式逐步更新特征，DenseNet 以拼接方式持续积累并复用特征；DenseNet 的优势在特征复用和参数效率，主要代价在激活存储与数据访问。**

---

## 4. 补充：数值复核

随作业提供 [verify_resnet_gradients.py](verify_resnet_gradients.py)，仅使用 Python 3 标准库，无需安装深度学习框架。运行：

~~~powershell
python verify_resnet_gradients.py
~~~

脚本会生成 [verification_resnet_results.json](verification_resnet_results.json)，并用中心差分复核：

- 两个连续非线性残差单元中，全部权重、偏置与初始输入的解析梯度；
- 从更深单元传来的梯度对前一单元权重梯度的影响；
- Highway 耦合门层的完整输入雅可比；
- $F(x)=-x$ 时总梯度可以为零的反例。

例子采用 $F_l(x)=\tanh(W_lx+b_l)$ 与平方误差损失，全部数值为固定的教学构造。该复核检查上述公式，不代表在图像数据集上训练过 ResNet、Highway Network 或 DenseNet。

## 参考文献

1. He, K., Zhang, X., Ren, S., & Sun, J. **Deep Residual Learning for Image Recognition**. CVPR, 2016. [论文](https://arxiv.org/abs/1512.03385)。用于核对残差学习、恒等快捷连接与投影。
2. He, K., Zhang, X., Ren, S., & Sun, J. **Identity Mappings in Deep Residual Networks**. ECCV, 2016. [论文](https://arxiv.org/abs/1603.05027)。用于核对前向/反向传播及恒等映射条件。
3. Srivastava, R. K., Greff, K., & Schmidhuber, J. **Highway Networks**. arXiv:1505.00387, 2015. [论文](https://arxiv.org/abs/1505.00387)。用于核对变换门、传递门及其耦合形式。
4. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. **Densely Connected Convolutional Networks**. CVPR, 2017. [论文](https://arxiv.org/abs/1608.06993)。用于核对拼接连接、增长率、瓶颈与压缩。
5. Pleiss, G., Chen, D., Huang, G., Li, T., van der Maaten, L., & Weinberger, K. Q. **Memory-Efficient Implementation of DenseNets**. arXiv:1707.06990, 2017. [论文](https://arxiv.org/abs/1707.06990)。用于核对训练显存开销与优化方案。

