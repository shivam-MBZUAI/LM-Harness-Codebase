# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""

from datasets import load_dataset
import os.path as osp

from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score
from lm_eval.utils import general_detokenize, camel_clean, ARA_DATA_DIR
from lm_eval import metrics
import evaluate

# from datasets import DatasetDict

_CITATION = """
"""


class SUMM_AR(Task):
    VERSION = 0

    rouge = evaluate.load('rouge')

    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'test_ar_summarization.jsonl')})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _process_doc(self, doc):
        source = camel_clean(doc["source"])
        target = camel_clean(doc["target"])
        out_doc = {
            "source": source,
            "target": target,
        }
        return out_doc

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["source"]

    def doc_to_target(self, doc):
        return doc["target"]

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["<|endoftext|>"]})
        return continuation

    def process_results(self, doc, results, save_all=False):
        completion = results[0].strip()
        refs = doc["target"]

        # Process sentence-level ROUGE for similarity measures.
        scores = self.rouge_score([completion], [refs])

        ref_pred = (refs, completion)

        result = {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeLsum": scores["rougeLsum"],
            "bleu": ref_pred,
        }

        if save_all:
            result["example"] = {
                "input": doc["source"],
                "ref": doc["target"],
                "pred": completion,
            }

        return result

    def aggregation(self):
        return {
            "rouge1": mean,
            "rouge2": mean,
            "rougeLsum": mean,
            "bleu": metrics.bleu,
        }

    def higher_is_better(self):
        return {
            "rouge1": True,
            "rouge2": True,
            "rougeLsum": True,
            "bleu": True,
        }

    def rouge_score(self, preds, refs):
        results = self.rouge.compute(predictions=preds,
                                     references=refs)

        return results


class SUMM_EN(SUMM_AR):
    def __init__(self, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("json", data_files={"test": osp.join(data_dir, 'test_en_summarization.jsonl')})

    def _process_doc(self, doc):
        return doc
