"""真实类别占比反推：在单纯形上解约束二次规划。

模型（类内稳定假设）：新批次预测分布 q ≈ X p
  - X: 校验集列条件概率矩阵（行=预测 j，列=真实 k）
  - p: 新批次真实类别占比，未知
  - q: 新批次预测标签的经验占比，已知

估计量：
    minimize_p  0.5 * ||X p - q||^2
    subject to  p_k >= 0,  sum(p) = 1

硬约束保证结果非负且总和为 1。**绝不先求逆/伪逆再截断**——
本模块代码路径中不存在任何矩阵求逆或线性最小二乘求解，
连兜底的投影梯度下降也只做梯度步 + 单纯形正交投影（排序+前缀和）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_SOLVER_TOL = 1e-12


def project_simplex(v: np.ndarray) -> np.ndarray:
    """把向量正交投影到单纯形 {p >= 0, sum(p) = 1}。

    Duchi et al. (2008) 的 O(n log n) 算法：仅排序与前缀和，无矩阵运算。
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # rho = 满足 u_j - (cssv_j - 1)/(j+1) > 0 的最大下标
    js = np.arange(1, n + 1)
    candidates = u - (cssv - 1.0) / js
    feasible = np.nonzero(candidates > 0)[0]
    if feasible.size == 0:  # 数值极端情况：均匀分布兜底
        return np.full(n, 1.0 / n)
    rho = feasible[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def _projected_gradient(
    X: np.ndarray, q: np.ndarray, max_iter: int = 200000, tol: float = 1e-12
) -> tuple[np.ndarray, dict]:
    """兜底求解器：投影梯度下降（加速 FISTA 风格），无求逆。"""
    k = X.shape[1]
    XtX = X.T @ X
    Xtq = X.T @ q
    # Lipschitz 常数（用幂迭代估计最大特征值，同样不做分解求逆）
    eig = np.ones(k)
    for _ in range(100):
        eig_new = XtX @ eig
        norm = np.linalg.norm(eig_new)
        if norm == 0:
            break
        eig = eig_new / norm
    lip = float(np.linalg.norm(XtX @ eig))
    step = 1.0 / lip if lip > 0 else 1.0

    x = project_simplex(np.full(k, 1.0 / k))
    y = x.copy()
    t = 1.0
    prev_obj = np.inf
    iters = 0
    for iters in range(1, max_iter + 1):
        grad = XtX @ y - Xtq
        x_new = project_simplex(y - step * grad)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        obj = 0.5 * float(np.sum((X @ x - q) ** 2))
        if abs(prev_obj - obj) < tol and iters > 50:
            break
        prev_obj = obj
    return x, {"method": "projected_gradient", "iterations": iters}


def estimate_true_proportions(
    X: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, dict]:
    """解单纯形约束 QP，返回 (p, 诊断信息)。

    p 保证 sum(p)=1 且 min(p)>=0（数值噪声在 1e-10 内归零后重新归一）。
    """
    X = np.asarray(X, dtype=float)
    q = np.asarray(q, dtype=float)
    if X.ndim != 2 or X.shape[0] != q.shape[0]:
        raise ValueError("X 与 q 形状不匹配")
    k = X.shape[1]
    if not np.isfinite(X).all() or not np.isfinite(q).all():
        raise ValueError("输入包含 NaN/Inf")

    x0 = project_simplex(np.full(k, 1.0 / k))
    constraints = ({"type": "eq", "fun": lambda p: np.sum(p) - 1.0},)
    bounds = [(0.0, 1.0)] * k

    info: dict = {}
    p = None
    try:
        result = minimize(
            lambda p: 0.5 * float(np.sum((X @ p - q) ** 2)),
            x0,
            jac=lambda p: X.T @ (X @ p - q),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        if result.success or (
            np.isfinite(result.x).all()
            and abs(result.x.sum() - 1.0) < 1e-6
            and result.x.min() > -1e-8
        ):
            p = result.x
            info["method"] = "SLSQP"
            info["iterations"] = int(result.nit) if result.nit else None
            info["message"] = str(result.message)
    except Exception as exc:  # pragma: no cover - 兜底路径
        info["slsqp_error"] = repr(exc)

    if p is None:
        p, pg_info = _projected_gradient(X, q)
        info.update(pg_info)

    # 数值清理：仅清理 ~1e-10 级噪声，不做业务截断
    p = np.where(p < 0, 0.0, p)
    p[p < 1e-10] = 0.0
    total = p.sum()
    if total <= 0:  # 理论上不可能，防御性处理
        p = project_simplex(np.full(k, 1.0 / k))
        total = p.sum()
    p = p / total

    residual = X @ p - q
    info["objective"] = 0.5 * float(residual @ residual)
    info["rmse"] = float(np.sqrt(np.mean(residual**2)))
    info["max_abs_residual"] = float(np.max(np.abs(residual)))
    info["sum"] = float(p.sum())
    info["min"] = float(p.min())
    return p, info
