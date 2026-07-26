"""Unit tests for evaluation metrics."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluation.metrics import coverage_at_acc, accuracy, is_correct


def test_is_correct_normalization():
    assert is_correct("  Paris ", "paris") is True
    assert is_correct("PARIS.", "paris") is True
    assert is_correct("London", "paris") is False


def test_coverage_at_acc_excludes_abstentions():
    # 2 answered (1 correct, 1 wrong), 1 abstained -> cov@acc = 1/2 = 0.5
    preds = ["paris", "london", "rome"]
    abstained = [False, False, True]
    golds = ["paris", "paris", "rome"]
    assert coverage_at_acc(preds, abstained, golds) == 0.5


def test_accuracy_counts_abstain_as_wrong():
    preds = ["paris", "london", "rome"]
    abstained = [False, False, True]
    golds = ["paris", "paris", "rome"]
    # 1 correct out of 3 (abstain counts as wrong)
    assert abs(accuracy(preds, abstained, golds) - 1/3) < 1e-9


def test_coverage_at_acc_all_abstain():
    # degenerate: everyone abstains -> 0/0, should not crash
    preds = ["a", "b"]
    abstained = [True, True]
    golds = ["a", "b"]
    # convention: 0 coverage@acc when nothing answered
    val = coverage_at_acc(preds, abstained, golds)
    assert val == 0.0
