import unittest
from run_deployment_search import eligible_models


class SelectionTests(unittest.TestCase):
    def test_negative_seed_and_closed_test(self):
        models = {"a": {}, "b": {}}
        per = [
            dict(model=m, stage="RAM_Cascade", split="validation", fit=50)
            for m in models
        ]
        per.append(dict(model="a", stage="RAM_Cascade", split="test", fit=-1000))
        candidates = [
            dict(
                model="b",
                kind="seed_flight",
                stage="RAM_Cascade",
                split="validation",
                fit=-1,
            )
        ]
        self.assertEqual(eligible_models(models, per, candidates), ["a"])
