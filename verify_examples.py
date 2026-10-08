"""复核作业中的 SVD、贝叶斯反演与交叉熵，并生成配图。

数值都是教学例子。本脚本不调用 Lumerical FDTD，也不生成真实实验数据。
运行：python verify_examples.py
"""

from pathlib import Path
import json
import math
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def close(actual, expected, label, atol=1e-10):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def integrate(values, grid):
    """梯形积分，兼容 NumPy 1.x 与 2.x。"""
    return float(np.sum((values[1:] + values[:-1]) * np.diff(grid) / 2))


def svd_example():
    a = np.array([[1., 1.], [1., 0.], [0., 1.]])
    u = np.array([
        [math.sqrt(2 / 3), 0., 1 / math.sqrt(3)],
        [1 / math.sqrt(6), 1 / math.sqrt(2), -1 / math.sqrt(3)],
        [1 / math.sqrt(6), -1 / math.sqrt(2), -1 / math.sqrt(3)],
    ])
    sigma = np.array([[math.sqrt(3), 0.], [0., 1.], [0., 0.]])
    v = np.array([[1., 1.], [1., -1.]]) / math.sqrt(2)
    close(a.T @ a, [[2., 1.], [1., 2.]], "A^T A")
    close(u.T @ u, np.eye(3), "U orthogonality")
    close(v.T @ v, np.eye(2), "V orthogonality")
    close(a @ v, u @ sigma, "A V = U Sigma")
    close(u @ sigma @ v.T, a, "SVD reconstruction")
    _, singular_values, _ = np.linalg.svd(a, full_matrices=True)
    close(singular_values, [math.sqrt(3), 1.], "NumPy singular values")
    rank_one = math.sqrt(3) * np.outer(u[:, 0], v[:, 0])
    close(rank_one, [[1., 1.], [.5, .5], [.5, .5]], "rank-one approximation")
    close(np.linalg.norm(a - rank_one, "fro"), 1., "rank-one error")
    return {
        "A": a.tolist(), "U": u.tolist(), "Sigma": sigma.tolist(), "V": v.tolist(),
        "singular_values": singular_values.tolist(),
        "max_reconstruction_error": float(np.max(np.abs(u @ sigma @ v.T - a))),
        "rank_one": rank_one.tolist(), "rank_one_frobenius_error": 1.,
        "rank_one_energy_fraction": .75,
    }


def bayes_example():
    prior_mean, prior_variance = 0., 100.
    lambda_0, sensitivity, noise_std = 1550., 2., 3.
    observations = np.array([1564., 1568., 1566.])
    residuals = observations - lambda_0
    n = len(observations)
    variance = 1 / (1 / prior_variance + n * sensitivity**2 / noise_std**2)
    mean = variance * (prior_mean / prior_variance + sensitivity * residuals.sum() / noise_std**2)
    std = math.sqrt(variance)
    close(mean, 3200 / 403, "posterior mean")
    close(variance, 300 / 403, "posterior variance")
    z_975 = NormalDist().inv_cdf(.975)
    credible_interval = [mean - z_975 * std, mean + z_975 * std]

    # 三次顺序更新与一次批量更新必须得到同一个后验。
    seq_mean, seq_var, updates = prior_mean, prior_variance, []
    for observation in observations:
        next_var = 1 / (1 / seq_var + sensitivity**2 / noise_std**2)
        seq_mean = next_var * (seq_mean / seq_var + sensitivity * (observation - lambda_0) / noise_std**2)
        seq_var = next_var
        updates.append({"observation_nm": float(observation), "mean_nm": seq_mean, "variance_nm2": seq_var})
    close(seq_mean, mean, "sequential mean")
    close(seq_var, variance, "sequential variance")

    # 用先验 × 似然的独立网格积分核对解析后验。
    grid = np.linspace(-40., 40., 40001)
    log_weights = -.5 * (grid - prior_mean)**2 / prior_variance
    log_weights -= .5 * np.sum((residuals[:, None] - sensitivity * grid[None, :])**2, axis=0) / noise_std**2
    weights = np.exp(log_weights - log_weights.max())
    integral = integrate(weights, grid)
    density = weights / integral
    numeric_mean = integrate(grid * density, grid)
    numeric_var = integrate((grid - numeric_mean)**2 * density, grid)
    close(numeric_mean, mean, "grid posterior mean", atol=1e-8)
    close(numeric_var, variance, "grid posterior variance", atol=1e-8)

    prediction_mean = lambda_0 + sensitivity * mean
    prediction_variance = noise_std**2 + sensitivity**2 * variance
    prediction_std = math.sqrt(prediction_variance)

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    prior_density = np.exp(-grid**2 / (2 * prior_variance)) / math.sqrt(2 * math.pi * prior_variance)
    posterior_density = np.exp(-(grid - mean)**2 / (2 * variance)) / math.sqrt(2 * math.pi * variance)
    ax.plot(grid, prior_density, color="#0072B2", lw=2.2, label="先验：N(0, 100)")
    ax.plot(grid, posterior_density, color="#D55E00", lw=2.2, label=f"后验：N({mean:.3f}, {variance:.3f})")
    inside = (grid >= credible_interval[0]) & (grid <= credible_interval[1])
    ax.fill_between(grid, 0, posterior_density, where=inside, color="#D55E00", alpha=.18, label="后验 95% 可信区间")
    ax.axvline(mean, color="#D55E00", ls=":", lw=1.3)
    ax.set(xlim=(-30, 30), ylim=(0, .50), xlabel="尺寸偏差 θ / nm", ylabel=r"概率密度 / $\mathrm{nm}^{-1}$", title="三次观测后，尺寸偏差的不确定性减小")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(ASSETS / "bayes_prior_posterior.png", dpi=180)
    fig.savefig(ASSETS / "bayes_prior_posterior.svg")
    plt.close(fig)

    return {
        "observations_nm": observations.tolist(), "sensitivity_nm_per_nm": sensitivity,
        "prior_mean_nm": prior_mean, "prior_variance_nm2": prior_variance,
        "posterior_mean_nm": mean, "posterior_variance_nm2": variance,
        "posterior_std_nm": std, "posterior_95_credible_interval_nm": credible_interval,
        "mle_nm": float(residuals.mean() / sensitivity), "map_nm": mean,
        "sequential_updates": updates,
        "grid_mean_nm": numeric_mean, "grid_variance_nm2": numeric_var,
        "next_observation_mean_nm": prediction_mean,
        "next_observation_variance_nm2": prediction_variance,
        "next_observation_95_prediction_interval_nm": [prediction_mean - z_975 * prediction_std, prediction_mean + z_975 * prediction_std],
    }


def entropy_example():
    labels = np.array([1., 0., 1.])
    probabilities = np.array([.8, .3, .6])
    losses = -(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
    mean_loss = float(losses.mean())
    close(np.exp(-losses.sum()), .336, "joint likelihood")
    close(mean_loss, -math.log(.336) / 3, "negative average log likelihood")
    p = np.array([.7, .2, .1])
    q = np.array([.6, .3, .1])
    h_p = float(-np.sum(p * np.log(p)))
    h_pq = float(-np.sum(p * np.log(q)))
    kl = float(np.sum(p * np.log(p / q)))
    close(h_pq, h_p + kl, "cross-entropy = entropy + KL")

    # 从 logit 出发，用中心差分独立复核 dL/dz = sigmoid(z) - y。
    logits = np.log(probabilities / (1 - probabilities))
    def scalar_bce(z, y):
        return float(np.logaddexp(0., z) - y * z)
    eps = 1e-5
    numeric_gradient = np.array([(scalar_bce(z + eps, y) - scalar_bce(z - eps, y)) / (2 * eps) for z, y in zip(logits, labels)])
    close(numeric_gradient, probabilities - labels, "BCE logit gradient", atol=1e-8)

    # 教学性的一步更新：三个独立 logits，平均损失，学习率 0.3。
    # 它展示损失梯度，不声称是共享参数网络的完整训练过程。
    learning_rate = .3
    logits_after = logits - learning_rate * (probabilities - labels) / len(labels)
    probabilities_after = 1 / (1 + np.exp(-logits_after))
    losses_after = np.array([scalar_bce(z, y) for z, y in zip(logits_after, labels)])
    if float(losses_after.mean()) >= mean_loss:
        raise AssertionError("One-step update must decrease the example loss")
    close(losses_after.mean(), .3539871960203372, "updated mean BCE")

    x = np.linspace(.005, .995, 1000)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    ax.plot(x, -np.log(x), color="#D55E00", lw=2.2, label="真实标签 y=1：−ln(q)")
    ax.plot(x, -np.log1p(-x), color="#0072B2", lw=2.2, label="真实标签 y=0：−ln(1−q)")
    ax.scatter(probabilities[labels == 1], losses[labels == 1], color="#D55E00", s=50, zorder=3)
    ax.scatter(probabilities[labels == 0], losses[labels == 0], color="#0072B2", marker="s", s=50, zorder=3)
    for qi, li, index in zip(probabilities, losses, [1, 2, 3]):
        ax.annotate(f"样本 {index}", (qi, li), xytext=(6, 8), textcoords="offset points", fontsize=10)
    ax.set(xlim=(0, 1), ylim=(0, 5.5), xlabel="模型对 y=1 的预测概率 q", ylabel="交叉熵损失 / nat", title="给真实类别的概率越低，交叉熵损失越大")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, loc="upper center")
    fig.savefig(ASSETS / "binary_cross_entropy.png", dpi=180)
    fig.savefig(ASSETS / "binary_cross_entropy.svg")
    plt.close(fig)

    return {
        "labels": labels.tolist(), "probabilities": probabilities.tolist(),
        "per_sample_loss_nats": losses.tolist(), "mean_loss_nats": mean_loss,
        "joint_likelihood": .336,
        "distribution_entropy_nats": h_p, "distribution_cross_entropy_nats": h_pq,
        "distribution_kl_nats": kl, "logit_gradients": numeric_gradient.tolist(),
        "independent_logits_step": {"learning_rate": learning_rate, "logits_before": logits.tolist(), "logits_after": logits_after.tolist(), "probabilities_after": probabilities_after.tolist(), "per_sample_loss_after_nats": losses_after.tolist(), "mean_loss_after_nats": float(losses_after.mean())},
    }


if __name__ == "__main__":
    results = {"svd": svd_example(), "bayes": bayes_example(), "cross_entropy": entropy_example()}
    (BASE / "verification_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("数值复核通过：SVD 重构与低秩误差；贝叶斯解析/顺序/网格后验；交叉熵似然与梯度。")
    print(f"贝叶斯后验：均值 {results['bayes']['posterior_mean_nm']:.6f} nm，标准差 {results['bayes']['posterior_std_nm']:.6f} nm")
    print(f"平均二分类交叉熵：{results['cross_entropy']['mean_loss_nats']:.6f} nat")
    print("生成：verification_results.json、assets/bayes_prior_posterior.png、assets/binary_cross_entropy.png")
