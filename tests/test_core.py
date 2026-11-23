"""核心数学性质单元测试，运行: python3 -I -m unittest discover -s tests -v"""

import inspect
import unittest

import numpy as np

from app.core.bootstrap import (
    bootstrap_proportions,
    marginal_fit_test,
    transition_matrix_test,
)
from app.core.confusion import build_confusion, predicted_distribution
from app.core.identifiability import diagnose_identifiability
from app.core.merge import merge_confusion
from app.core.pipeline import run_analysis
from app.core.solver import estimate_true_proportions, project_simplex
from app.core.synthetic import (
    BASELINE_X,
    DRIFT_X,
    DRIFT_TRUE_P,
    generate_synthetic_scenario,
)


def _simulate(true_p, X, n, rng):
    k = len(true_p)
    yt = rng.choice(k, size=n, p=true_p)
    yp = np.empty(n, dtype=int)
    for kk in range(k):
        mask = yt == kk
        m = int(mask.sum())
        if m:
            yp[mask] = rng.choice(k, size=m, p=X[:, kk])
    return yt, yp


class TestSolver(unittest.TestCase):
    def test_simplex_and_nonnegativity(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            X = rng.dirichlet(np.ones(5), size=5).T  # 列随机
            p_true = rng.dirichlet(np.ones(5))
            q = X @ p_true
            p_hat, info = estimate_true_proportions(X, q)
            self.assertAlmostEqual(p_hat.sum(), 1.0, places=10)
            self.assertGreaterEqual(p_hat.min(), 0.0)
            self.assertTrue(np.isfinite(p_hat).all())

    def test_recovers_known_mixture(self):
        rng = np.random.default_rng(1)
        X = BASELINE_X
        p_true = np.array([0.4, 0.3, 0.2, 0.1])
        q = X @ p_true
        p_hat, _ = estimate_true_proportions(X, q)
        np.testing.assert_allclose(p_hat, p_true, atol=1e-6)

    def test_no_inversion_in_estimate_path(self):
        """反推路径不允许出现求逆/伪逆/线性求解后截断的写法。"""
        import ast

        import app.core.solver as solver_mod

        tree = ast.parse(inspect.getsource(solver_mod))
        banned_funcs = {"inv", "pinv", "solve", "lstsq", "normal"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned_funcs:
                self.fail(f"求解器中禁止调用 np.linalg.{node.attr}")
            if isinstance(node, ast.Name) and node.id in banned_funcs:
                self.fail(f"求解器中禁止调用 {node.id}")

    def test_project_simplex_math(self):
        v = np.array([3.0, -1.0, 0.5, 2.0])
        w = project_simplex(v)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertGreaterEqual(w.min(), 0.0)
        # 已在单纯形内的点投影后不变
        u = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(project_simplex(u), u)

    def test_zero_column_does_not_crash(self):
        X = np.array([[0.8, 0.0], [0.2, 1.0]])  # 真实类 0 无校验样本（零列）
        q = np.array([0.4, 0.6])
        p_hat, info = estimate_true_proportions(X, q)
        self.assertAlmostEqual(p_hat.sum(), 1.0)
        self.assertGreaterEqual(p_hat.min(), 0.0)


class TestConfusion(unittest.TestCase):
    def test_column_stochastic_and_orientation(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "a", "b"]
        cm = build_confusion(y_true, y_pred, labels=["a", "b"])
        np.testing.assert_allclose(cm.matrix.sum(axis=0), [1.0, 1.0])
        self.assertEqual(cm.counts[0, 1], 1.0)  # 真 b 预 a
        self.assertEqual(cm.col_support.tolist(), [2, 2])

    def test_label_only_in_prediction_kept(self):
        cm = build_confusion(["a"], ["b"], labels=["a", "b"])
        self.assertEqual(cm.col_support.tolist(), [1, 0])  # b 为零列但保留


class TestIdentifiability(unittest.TestCase):
    def test_missing_class_flagged(self):
        X = np.array([[0.8, 0.0], [0.2, 1.0]])
        diag = diagnose_identifiability(["a", "b"], X, np.array([5, 0]))
        self.assertIn("b", diag["missing_classes"])
        self.assertTrue(any(g["missing"] for g in diag["groups"]))

    def test_near_identical_columns_grouped(self):
        X = np.array([[0.70, 0.701], [0.30, 0.299]])
        diag = diagnose_identifiability(
            ["x", "y"], X, np.array([100, 100]), similarity_threshold=0.98
        )
        self.assertEqual(len(diag["near_identical_pairs"]), 1)
        ambiguous = [g for g in diag["groups"] if len(g["members"]) > 1]
        self.assertEqual(ambiguous[0]["members"], ["x", "y"])
        self.assertTrue(diag["warnings"])

    def test_well_separated_columns_no_warning(self):
        X = np.array([[0.95, 0.05], [0.05, 0.95]])
        diag = diagnose_identifiability(["x", "y"], X, np.array([100, 100]))
        self.assertEqual(diag["near_identical_pairs"], [])
        self.assertFalse(diag["rank_deficient"])


class TestMerge(unittest.TestCase):
    def test_merge_preserves_totals_and_originals(self):
        rng = np.random.default_rng(3)
        yt_i, yp_i = _simulate(
            np.array([0.5, 0.3, 0.2]),
            np.array([[0.8, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.7]]),
            2000,
            rng,
        )
        labels = ["a", "b", "c"]
        yt = [labels[i] for i in yt_i]
        yp = [labels[i] for i in yp_i]
        cm = build_confusion(yt, yp, labels=labels)
        merged, plan = merge_confusion(cm, [["b", "c"]])
        self.assertEqual(merged.n, cm.n)
        np.testing.assert_allclose(merged.matrix.sum(axis=0), 1.0)
        self.assertEqual(plan["original_labels"], labels)  # 合并前定义保留
        self.assertEqual(merged.labels, ["b+c", "a"])

    def test_merge_rejects_duplicate_membership(self):
        cm = build_confusion(["a", "b", "c"], ["a", "b", "c"], labels=["a", "b", "c"])
        with self.assertRaises(ValueError):
            merge_confusion(cm, [["a", "b"], ["b", "c"]])


class TestBootstrapAndResiduals(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.labels = ["咨询", "投诉", "查询进度", "表扬"]
        yt_i, yp_i = _simulate(
            np.array([0.55, 0.2, 0.2, 0.05]), BASELINE_X, 4000, self.rng
        )
        self.y_true = [self.labels[i] for i in yt_i]
        self.y_pred = [self.labels[i] for i in yp_i]
        self.cm = build_confusion(self.y_true, self.y_pred, labels=self.labels)

    def test_bootstrap_intervals_cover_truth(self):
        q = BASELINE_X @ DRIFT_TRUE_P
        boot = bootstrap_proportions(
            np.array(self.y_true),
            np.array(self.y_pred),
            self.labels,
            q,
            n_replicates=200,
            rng=np.random.default_rng(7),
        )
        lo = np.array([v[0] for v in boot["intervals"]["95"]])
        hi = np.array([v[1] for v in boot["intervals"]["95"]])
        self.assertEqual(boot["n_valid"] + boot["skipped_zero_column"], 200)
        covered = (DRIFT_TRUE_P >= lo) & (DRIFT_TRUE_P <= hi)
        self.assertGreaterEqual(covered.sum(), 3)  # 4 类至少 3 类覆盖

    def test_bootstrap_survives_structural_zero_column(self):
        # 结构性缺席列（原校验集就没有该真值类别）不应导致每次重抽都被跳过。
        labels3 = ["咨询", "投诉", "查询进度"]
        rng = np.random.default_rng(11)
        X3 = BASELINE_X[:3, :3].copy()
        X3 = X3 / X3.sum(axis=0, keepdims=True)  # 截成 3 类后重新列归一化
        yt_i, yp_i = _simulate(np.array([0.5, 0.3, 0.2]), X3, 1200, rng)
        y_true = [labels3[i] for i in yt_i]
        y_pred = [labels3[i] for i in yp_i]
        # 人为制造一个零列类别"表扬"：仅在预测侧出现，真值侧永不出现
        labels4 = labels3 + ["表扬"]
        y_true4 = np.array(y_true)
        y_pred4 = np.array(
            [v if i % 40 else "表扬" for i, v in enumerate(y_pred)]
        )
        q = np.array([0.45, 0.3, 0.2, 0.05])
        boot = bootstrap_proportions(
            y_true4, y_pred4, labels4, q,
            n_replicates=100, rng=np.random.default_rng(12),
        )
        # 修复前这里会 RuntimeError；现在绝大多数重抽应有效
        self.assertGreaterEqual(boot["n_valid"], 95)
        med = np.array(boot["median"])
        self.assertTrue(np.isfinite(med).all())
        # 缺席类别的估计仅受单纯形约束，区间仍非负
        lo = np.array([v[0] for v in boot["intervals"]["95"]])
        self.assertTrue((lo >= -1e-12).all())

    def test_marginal_test_reports_saturation(self):
        # 方阵满秩下边际 GOF 饱和、无功效，模块必须如实标注而非假装检出
        q = BASELINE_X @ DRIFT_TRUE_P
        p_hat, _ = estimate_true_proportions(BASELINE_X, q)
        test = marginal_fit_test(
            BASELINE_X, q, p_hat, 3000, n_replicates=200,
            rng=np.random.default_rng(8),
        )
        self.assertTrue(test["saturated_no_power"])
        self.assertLess(test["statistic_t2"], 1e-6)

    def test_transition_matrix_test_distinguishes_a_and_b(self):
        # 带真值的两样本检验：A 同机制不漂移，B 类内漂移应被逐列 TV 识别
        rng = np.random.default_rng(2024)
        labels = self.labels
        val_true_i, val_pred_i = _simulate(
            np.array([0.55, 0.2, 0.2, 0.05]), BASELINE_X, 4000, rng
        )
        val_true = [labels[i] for i in val_true_i]
        val_pred = [labels[i] for i in val_pred_i]

        def new_matrix(mechanism):
            bt_i, bp_i = _simulate(DRIFT_TRUE_P, mechanism, 3000, rng)
            bt_labels = [labels[i] for i in bt_i]
            cm_new = build_confusion(
                bt_labels,
                [labels[i] for i in bp_i],
                labels=labels,
            )
            return cm_new.matrix, bt_labels

        mat_a, true_a = new_matrix(BASELINE_X)
        test_a = transition_matrix_test(
            BASELINE_X, np.array(val_true), np.array(val_pred),
            mat_a, np.array(true_a), labels,
            n_replicates=300, rng=np.random.default_rng(8),
        )
        mat_b, true_b = new_matrix(DRIFT_X)
        test_b = transition_matrix_test(
            BASELINE_X, np.array(val_true), np.array(val_pred),
            mat_b, np.array(true_b), labels,
            n_replicates=300, rng=np.random.default_rng(9),
        )
        self.assertFalse(test_a["misfit_flagged"])
        self.assertTrue(test_b["misfit_flagged"])
        self.assertGreater(test_b["overall_max_tv"], test_a["overall_max_tv"] * 3)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_synthetic_scenarios(self):
        payload = generate_synthetic_scenario()
        cm = build_confusion(
            payload["validation"]["y_true"],
            payload["validation"]["y_pred"],
            labels=payload["labels"],
        )
        for key in ("scenario_a", "scenario_b"):
            sc = payload[key]
            result = run_analysis(
                cm, sc["pred_counts"],
                y_true=payload["validation"]["y_true"],
                y_pred=payload["validation"]["y_pred"],
                batch_true_counts=sc["true_counts"],
                batch_y_true=sc["y_true"],
                batch_y_pred=sc["y_pred"],
                n_bootstrap=100, n_null=300,
            )
            p = np.array(result["estimate"]["p"])
            self.assertAlmostEqual(p.sum(), 1.0, places=9)
            self.assertGreaterEqual(p.min(), 0.0)
            self.assertTrue(result["estimate"]["constraints_ok"])
            self.assertFalse(result["estimate"]["inverse_used"])
            self.assertIn("bootstrap", result)
            # 估计占比正向推回与观测 q 的 TV 距离应很小（QP 最优性）
            self.assertLess(result["forward_check"]["total_variation_distance"], 0.05)

        # 配对序列修复后：A 同机制不应误报，B 机制漂移应被识别
        r_a = run_analysis(
            cm, payload["scenario_a"]["pred_counts"],
            y_true=payload["validation"]["y_true"],
            y_pred=payload["validation"]["y_pred"],
            batch_y_true=payload["scenario_a"]["y_true"],
            batch_y_pred=payload["scenario_a"]["y_pred"],
            n_bootstrap=50, n_null=300,
        )
        r_b = run_analysis(
            cm, payload["scenario_b"]["pred_counts"],
            y_true=payload["validation"]["y_true"],
            y_pred=payload["validation"]["y_pred"],
            batch_y_true=payload["scenario_b"]["y_true"],
            batch_y_pred=payload["scenario_b"]["y_pred"],
            n_bootstrap=50, n_null=300,
        )
        self.assertFalse(r_a["transition_test"]["misfit_flagged"])
        self.assertTrue(r_b["transition_test"]["misfit_flagged"])

    def test_predicted_distribution_validation(self):
        with self.assertRaises(ValueError):
            predicted_distribution({"未知类": 1}, ["a", "b"])
        q = predicted_distribution({"a": 3, "b": 1}, ["a", "b"])
        np.testing.assert_allclose(q, [0.75, 0.25])


if __name__ == "__main__":
    unittest.main()
