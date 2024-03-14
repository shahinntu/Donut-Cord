import numpy as np
from nltk import edit_distance
from donut import JSONParseEvaluator


def edit_similarity(predictions, targets):
    scores = []
    for pred, target in zip(predictions, targets):
        scores.append(edit_distance(pred, target) / max(len(pred), len(target)))

    return 1.0 - np.mean(scores)


class Accuracy:
    def __init__(self, processor):
        self._processor = processor

        self._evaluator = JSONParseEvaluator()

    def __call__(self, predictions, targets):
        scores = []
        for pred, tgt in zip(predictions, targets):
            pred_json = self._processor.token2json(pred)
            tgt_json = self._processor.token2json(tgt)

            scores.append(self._evaluator.cal_acc(pred_json, tgt_json))

        return np.mean(scores)


class F1Score:
    def __init__(self, processor):
        self._processor = processor

        self._evaluator = JSONParseEvaluator()

    def __call__(self, predictions, targets):
        pred_jsons = [self._processor.token2json(pred) for pred in predictions]
        tgt_jsons = [self._processor.token2json(tgt) for tgt in targets]

        return self._evaluator.cal_f1(pred_jsons, tgt_jsons)
