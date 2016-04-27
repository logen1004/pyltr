"""

Testing for R2 metric.

"""

import pyltr


def test():
    m = pyltr.metrics.R2()

    # perfect prediction
    assert 1.0 == m.evaluate_preds(None, [0, 1, 2], [0, 1, 2])

    # terrible prediction
    assert -3.0 == m.evaluate_preds(None, [1, 2, 3, 4, 5], [5, 4, 3, 2, 1])

    # zero effort prediction
    assert 0.0 == m.evaluate_preds(None, [1, 2, 3, 4, 5], [3, 3, 3, 3, 3])
