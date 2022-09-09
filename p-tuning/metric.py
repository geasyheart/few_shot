# -*- coding: utf8 -*-
#
from typing import Dict

from utrainer.metric import Metric


class ACCMetric(Metric):
    def __init__(self):
        self.corr = 0
        self.all = 0

    def step(self, inputs):
        preds, trues = inputs
        for p, t in zip(preds, trues):
            self.all += 1
            if p == t:
                self.corr += 1

    def score(self) -> float:
        return self.corr / (self.all or 1e-5)

    def report(self) -> Dict:
        print(self.score())
        return {}
