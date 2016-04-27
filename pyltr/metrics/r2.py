"""

Expected reciprocal rank.

TODO: better docs

"""

import numpy as np
from . import Metric
from overrides import overrides
from sklearn.externals.six.moves import range
from sklearn.metrics import r2_score


class R2(Metric):
    """Optimizes for R2; the corresponding loss is squared error.

    """
    is_ltr_metric = False

    def __init__(self):
        super(R2, self).__init__()

    @overrides
    def calc_lambdas_deltas(self, qid, targets, preds):
        ns = targets.shape[0]

        lambdas = targets - preds  # negative derivative is just residual
        deltas = np.ones(ns)

        return lambdas, deltas

    @overrides
    def evaluate_preds(self, qid, targets, preds):
        assert qid is None, 'can\'t evaluate R2 on individual query'
        return r2_score(targets, preds)

    @overrides
    def calc_mean(self, qids, targets, preds):
        # queries don't matter, just calculate entire R2
        return r2_score(targets, preds)

    @overrides
    def calc_mean_random(self, qids, targets):
        # best constant estimator yields 0 R2
        return 0.0
