"""Tests for metrics."""

import pytest

from humanizer_bench.metrics import (
    accuracy,
    bootstrap_ci,
    f1_score,
    false_positive_rate,
    fpr_at_tpr,
    precision,
    recall,
    roc_auc,
)


def test_accuracy_perfect():
    assert accuracy([0, 1, 1], [0, 1, 1]) == 1.0


def test_accuracy_half():
    assert accuracy([0, 1], [0, 0]) == 0.5


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_false_positive_rate():
    # Human (label 0) examples are at indices 0 and 2; one is flagged -> 0.5.
    assert false_positive_rate([0, 1, 0], [1, 1, 0]) == 0.5


def test_false_positive_rate_no_negatives():
    assert false_positive_rate([1, 1], [1, 0]) == 0.0


def test_precision_half():
    # TP=1 (idx 1), FP=1 (idx 0), FN=1 (idx 2) -> 1/(1+1) = 0.5.
    assert precision([0, 1, 1, 0], [1, 1, 0, 0]) == 0.5


def test_precision_perfect():
    assert precision([0, 1, 1], [0, 1, 1]) == 1.0


def test_precision_no_predictions():
    assert precision([0, 1], [0, 0]) == 0.0


def test_precision_empty():
    assert precision([], []) == 0.0


def test_recall_half():
    # TP=1 (idx 1), FN=1 (idx 2) -> 1/(1+1) = 0.5.
    assert recall([0, 1, 1, 0], [1, 1, 0, 0]) == 0.5


def test_recall_perfect():
    assert recall([0, 1, 1], [0, 1, 1]) == 1.0


def test_recall_no_positives():
    assert recall([0, 0], [0, 1]) == 0.0


def test_f1_half():
    # precision = recall = 0.5 -> F1 = 0.5.
    assert f1_score([0, 1, 1, 0], [1, 1, 0, 0]) == 0.5


def test_f1_perfect():
    assert f1_score([0, 1, 1], [0, 1, 1]) == 1.0


def test_f1_all_wrong():
    assert f1_score([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0


def test_roc_auc_known_value():
    # Sorted: 0.1(neg,r1) 0.35(pos,r2) 0.4(neg,r3) 0.8(pos,r4).
    # rank_sum_pos = 6 -> (6 - 2*3/2) / (2*2) = 0.75.
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_roc_auc_perfect():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_roc_auc_inverted():
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_ties():
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_single_class():
    assert roc_auc([1, 1], [0.2, 0.8]) == 0.5
    assert roc_auc([0, 0], [0.2, 0.8]) == 0.5


def test_roc_auc_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="only 0/1 labels"):
        roc_auc([0, 2, 1], [0.1, 0.9, 0.5])


def test_bootstrap_ci_deterministic():
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]
    lo1, hi1 = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=500, seed=42)
    lo2, hi2 = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=500, seed=42)
    assert (lo1, hi1) == (lo2, hi2)
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_bootstrap_ci_perfect_predictor():
    y_true = [0, 1, 1, 0]
    lo, hi = bootstrap_ci(accuracy, y_true, y_true, n_bootstrap=200, seed=7)
    assert (lo, hi) == (1.0, 1.0)


def test_bootstrap_ci_roc_auc_with_scores():
    # Perfect ranking on a large enough sample: single-class resamples are
    # astronomically unlikely, so every resample scores 1.0.
    y_true = [0] * 10 + [1] * 10
    scores = [0.1 + 0.01 * i for i in range(10)] + [0.8 + 0.01 * i for i in range(10)]
    lo, hi = bootstrap_ci(roc_auc, y_true, scores, n_bootstrap=200, seed=3)
    assert (lo, hi) == (1.0, 1.0)


def test_bootstrap_ci_empty():
    assert bootstrap_ci(accuracy, [], [], n_bootstrap=100, seed=0) == (0.0, 0.0)


def test_bootstrap_ci_rejects_nonpositive_n_bootstrap():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 1, 0, 0]
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=0, seed=0)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=-5, seed=0)


def test_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci(accuracy, [0, 1, 1], [1, 1], n_bootstrap=100, seed=0)


def test_bootstrap_ci_rejects_invalid_ci():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 1, 0, 0]
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=100, ci=0.0, seed=0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=100, ci=1.5, seed=0)


def test_fpr_at_tpr_perfect_separation():
    # Every positive scores above every negative: catching 95% of AI texts
    # flags no human text.
    assert fpr_at_tpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_fpr_at_tpr_worst_separation():
    # Inverted ranking: to catch 95% of AI texts every human text is flagged.
    assert fpr_at_tpr([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_fpr_at_tpr_interpolates_and_groups_ties():
    # Sorted desc: 0.9(pos), then a tie group 0.5(pos)+0.5(neg).
    # ROC points: (0,0) -> (0,0.5) -> (1.0,1.0). At TPR=0.95 the straddling
    # segment interpolates to FPR = 0.9. A per-sample walk would wrongly cut
    # the tie group and report 0.0 -- no threshold can split tied scores.
    assert fpr_at_tpr([0, 1, 1], [0.5, 0.5, 0.9]) == pytest.approx(0.9)


def test_fpr_at_tpr_custom_tpr():
    # Sorted desc: 0.9(neg), 0.8(pos), 0.4(neg), 0.3(pos).
    # ROC points: (0,0) -> (0.5,0) -> (0.5,0.5) -> (1.0,0.5) -> (1.0,1.0).
    # At TPR=0.5 the curve passes exactly through (0.5, 0.5).
    assert fpr_at_tpr([0, 0, 1, 1], [0.9, 0.4, 0.8, 0.3], tpr=0.5) == 0.5


def test_fpr_at_tpr_degenerate_inputs():
    assert fpr_at_tpr([1, 1], [0.2, 0.8]) == 0.0  # no negatives
    assert fpr_at_tpr([0, 0], [0.2, 0.8]) == 0.0  # no positives
    assert fpr_at_tpr([], []) == 0.0  # empty


def test_fpr_at_tpr_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="only 0/1 labels"):
        fpr_at_tpr([0, 2, 1], [0.1, 0.9, 0.5])


def test_fpr_at_tpr_rejects_bad_tpr():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="strictly between"):
            fpr_at_tpr([0, 1], [0.2, 0.8], tpr=bad)
