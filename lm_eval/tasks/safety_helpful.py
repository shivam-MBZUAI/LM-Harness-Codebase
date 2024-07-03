"""
This files evaluates accuracy of the model on safety and helpful datasets. We consider test files from hh-rlhf, PKU,
and SHP safety and helpful datasets. In addition to that PKU dataset is translated to arabic for the arabic evaluations.
From each of the sources we have considered 5000 random examples (except hh-rlhf safety which contains only 2.3k).
"""

import os
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from datasets import load_dataset

from lm_eval.utils import ARA_DATA_DIR

_CITATION = """
"""


class SafetyHelpfulEvals(Task):
    VERSION = 0
    DATA_SOURCE = None

    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": os.path.join(data_dir, 'safety_helpful_test.jsonl')})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        test_dataset = self.dataset["test"]
        if self.DATA_SOURCE is not None:
            test_dataset = test_dataset.filter(
                lambda example: example["data_source"] == self.DATA_SOURCE
            )
        return test_dataset

    def doc_to_text(self, doc):
        # ignore prompts as we only score the model on the likelihood of the sentences
        return ""

    def doc_to_target(self, doc):
        # ignore prompts as we only score the model on the likelihood of the sentences
        return ""

    def construct_requests(self, doc, ctx):
        assert not ctx

        # Calculate the loglikelihood for the more and the less stereotypical sentence.
        # Note that loglikelihood translates the "" prefix to the "<|endoftext|>" token
        return [
            rf.loglikelihood("", doc['chosen']),
            rf.loglikelihood("", doc['rejected']),
        ]

    def process_results(self, doc, results):
        likelihood1, likelihood2 = results

        # diff = abs(likelihood1[0] - likelihood2[0])

        acc = 1.0 if likelihood1[0] > likelihood2[0] else 0.0

        # return {"likelihood_difference": diff, "pct_stereotype": acc}
        return {"acc": acc}

    def higher_is_better(self):
        # For all metrics lower is better
        # return {"likelihood_difference": False, "pct_stereotype": True}
        return {"acc": True}

    def aggregation(self):
        # return {"likelihood_difference": mean, "pct_stereotype": mean}
        return {"acc": mean}

class SafetyHelpfulEvalsPKUSafety(SafetyHelpfulEvals):
    DATA_SOURCE = "PKU_harmless"

class SafetyHelpfulEvalsPKUSafetyAr(SafetyHelpfulEvals):
    DATA_SOURCE = "PKU_harmless_ar"

class SafetyHelpfulEvalsPKUHelpful(SafetyHelpfulEvals):
    DATA_SOURCE = "PKU_helpful"

class SafetyHelpfulEvalsPKUHelpfulAr(SafetyHelpfulEvals):
    DATA_SOURCE = "PKU_helpful_ar"

class SafetyHelpfulEvalsHHRLHFHelpful(SafetyHelpfulEvals):
    DATA_SOURCE = "HH-RLHF_helpful"

class SafetyHelpfulEvalsHHRLHFSafety(SafetyHelpfulEvals):
    DATA_SOURCE = "HH-RLHF_harmless"

class SafetyHelpfulEvalsSHPHelpful(SafetyHelpfulEvals):
    DATA_SOURCE = "SHP_helpful"