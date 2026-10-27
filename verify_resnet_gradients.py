"""Finite-difference checks for the ResNet homework; Python 3 standard library only.

Run: python verify_resnet_gradients.py
The JSON output is written beside this file. No training data or network is needed.
"""

import argparse
import copy
import json
import math
from pathlib import Path


EPSILON = 1e-6
TOLERANCE = 1e-7

# Two nonlinear residual units with independent parameters.
LAYERS = [
    {
        "W": [[0.20, -0.10, 0.05], [0.30, 0.15, -0.20], [-0.10, 0.25, 0.10]],
        "b": [0.02, -0.03, 0.04],
    },
    {
        "W": [[-0.15, 0.20, 0.10], [0.05, -0.25, 0.30], [0.20, 0.10, -0.05]],
        "b": [-0.01, 0.02, -0.02],
    },
]
INPUT = [0.40, -0.70, 0.20]
TARGET = [0.10, -0.20, 0.50]


def affine(matrix, vector, bias):
    return [
        sum(weight * value for weight, value in zip(row, vector)) + shift
        for row, shift in zip(matrix, bias)
    ]


def forward(layers, initial):
    state = list(initial)
    cache = []
    for layer in layers:
        residual = [math.tanh(v) for v in affine(layer["W"], state, layer["b"])]
        cache.append((list(state), residual))
        state = [x + f for x, f in zip(state, residual)]
    return state, cache


def loss(layers, initial):
    output, _ = forward(layers, initial)
    return 0.5 * sum((value - target) ** 2 for value, target in zip(output, TARGET))


def backward(layers, initial):
    output, cache = forward(layers, initial)
    upstream = [value - target for value, target in zip(output, TARGET)]
    gradients = [None] * len(layers)
    feature_gradients = [None] * (len(layers) + 1)
    feature_gradients[-1] = list(upstream)
    for index in reversed(range(len(layers))):
        state, residual = cache[index]
        # delta = g_(l+1) elementwise multiplied by tanh'(W_l x_l + b_l)
        delta = [g * (1.0 - f * f) for g, f in zip(upstream, residual)]
        gradients[index] = {
            "W": [[d * x for x in state] for d in delta],
            "b": delta,
        }
        # The direct term is upstream, and the residual term is W_l^T delta.
        upstream = [
            upstream[j] + sum(layers[index]["W"][i][j] * delta[i] for i in range(3))
            for j in range(3)
        ]
        feature_gradients[index] = list(upstream)
    return gradients, feature_gradients


def summarize(analytical, numerical):
    errors = [abs(a - n) for a, n in zip(analytical, numerical)]
    return {
        "count": len(errors),
        "analytical": analytical,
        "finite_difference": numerical,
        "max_absolute_error": max(errors, default=0.0),
        "passed": all(math.isfinite(e) and e < TOLERANCE for e in errors),
    }


def check_residual_chain():
    gradients, feature_gradients = backward(LAYERS, INPUT)
    checks = []
    for layer_index in range(len(LAYERS)):
        analytical_w, numerical_w = [], []
        analytical_b, numerical_b = [], []
        for i in range(3):
            for j in range(3):
                plus, minus = copy.deepcopy(LAYERS), copy.deepcopy(LAYERS)
                plus[layer_index]["W"][i][j] += EPSILON
                minus[layer_index]["W"][i][j] -= EPSILON
                numerical_w.append((loss(plus, INPUT) - loss(minus, INPUT)) / (2 * EPSILON))
                analytical_w.append(gradients[layer_index]["W"][i][j])
            plus, minus = copy.deepcopy(LAYERS), copy.deepcopy(LAYERS)
            plus[layer_index]["b"][i] += EPSILON
            minus[layer_index]["b"][i] -= EPSILON
            numerical_b.append((loss(plus, INPUT) - loss(minus, INPUT)) / (2 * EPSILON))
            analytical_b.append(gradients[layer_index]["b"][i])
        checks.append({"name": f"resnet_unit_{layer_index}_weights", **summarize(analytical_w, numerical_w)})
        checks.append({"name": f"resnet_unit_{layer_index}_biases", **summarize(analytical_b, numerical_b)})

    numerical_input = []
    for j in range(3):
        plus, minus = list(INPUT), list(INPUT)
        plus[j] += EPSILON
        minus[j] -= EPSILON
        numerical_input.append((loss(LAYERS, plus) - loss(LAYERS, minus)) / (2 * EPSILON))
    checks.append({
        "name": "resnet_initial_input",
        **summarize(feature_gradients[0], numerical_input),
    })

    # Independently differentiate the suffix loss with respect to x_1.
    # This verifies the upstream g_1 used in the first unit's weight gradient.
    intermediate, _ = forward(LAYERS[:1], INPUT)
    numerical_upstream = []
    for j in range(3):
        plus, minus = list(intermediate), list(intermediate)
        plus[j] += EPSILON
        minus[j] -= EPSILON
        numerical_upstream.append(
            (loss(LAYERS[1:], plus) - loss(LAYERS[1:], minus)) / (2 * EPSILON)
        )
    checks.append({
        "name": "upstream_gradient_from_second_unit",
        **summarize(feature_gradients[1], numerical_upstream),
    })
    return checks, feature_gradients


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def highway(x):
    transform_weights = LAYERS[0]["W"]
    gate_weights = LAYERS[1]["W"]
    h = [math.tanh(v) for v in affine(transform_weights, x, LAYERS[0]["b"])]
    gate = [sigmoid(v) for v in affine(gate_weights, x, [-0.8, -1.0, -0.6])]
    output = [hv * tv + xv * (1.0 - tv) for hv, tv, xv in zip(h, gate, x)]
    return output, h, gate


def check_highway():
    _, h, gate = highway(INPUT)
    analytical = [
        [
            (1.0 - gate[i]) * float(i == j)
            + gate[i] * (1.0 - h[i] ** 2) * LAYERS[0]["W"][i][j]
            + (h[i] - INPUT[i]) * gate[i] * (1.0 - gate[i]) * LAYERS[1]["W"][i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]
    numerical = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        plus, minus = list(INPUT), list(INPUT)
        plus[j] += EPSILON
        minus[j] -= EPSILON
        out_plus, _, _ = highway(plus)
        out_minus, _, _ = highway(minus)
        for i in range(3):
            numerical[i][j] = (out_plus[i] - out_minus[i]) / (2 * EPSILON)
    return {
        "name": "highway_complete_input_jacobian",
        **summarize(
            [v for row in analytical for v in row],
            [v for row in numerical for v in row],
        ),
    }


def check_cancellation():
    # This intentionally uses a linear F(x) = -x, not the tanh units above.
    def constant_loss(x):
        output = [v + (-v) for v in x]
        return 0.5 * sum((v - t) ** 2 for v, t in zip(output, TARGET))

    numerical = []
    for j in range(3):
        plus, minus = list(INPUT), list(INPUT)
        plus[j] += EPSILON
        minus[j] -= EPSILON
        numerical.append((constant_loss(plus) - constant_loss(minus)) / (2 * EPSILON))
    return {
        "name": "identity_and_residual_can_cancel",
        "reason": "F(x)=-x gives I+J_F=0 even though the shortcut is identity.",
        **summarize([0.0] * 3, numerical),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("verification_resnet_results.json"),
    )
    args = parser.parse_args()
    checks, feature_gradients = check_residual_chain()
    checks.extend([check_highway(), check_cancellation()])
    passed = all(check["passed"] for check in checks)
    result = {
        "example": "Two 3D residual units F_l(x)=tanh(W_l x+b_l), squared-error loss.",
        "method": "Analytical derivatives versus central finite differences.",
        "epsilon": EPSILON,
        "absolute_tolerance": TOLERANCE,
        "layers": LAYERS,
        "input": INPUT,
        "target": TARGET,
        "loss": loss(LAYERS, INPUT),
        "feature_gradients_g0_g1_g2": feature_gradients,
        "checks": checks,
        "all_passed": passed,
        "scope": "Teaching examples only; no image dataset training or benchmark claims.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status} {check['name']}: max_abs_error={check['max_absolute_error']:.3e}")
    print(f"Results: {args.output}")
    if not passed:
        raise SystemExit("At least one gradient check failed.")


if __name__ == "__main__":
    main()

