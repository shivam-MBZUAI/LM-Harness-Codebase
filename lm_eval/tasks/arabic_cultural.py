from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score
import numpy as np
from datasets import load_dataset


class ArabicCultural(Task):
    VERSION = 0
    DATASET_NAME = "arabic_cultural"
    DATASET_PATH = "FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment"

    def __init__(self, cache_dir=None, download_mode=None):
        self.download(self.DATASET_PATH, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None


    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset(data_dir)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True
    

    def doc_to_text(self, doc):
        return 'الرجاء تحديد ما إذا كانت العبارة التالية صحيحة أم خاطئة. إذا كان صحيحاً، يرجى الرد بـ "نعم"، وإذا كان خطأ، يرجى الرد بـ "لا"' + \
                '\n' + doc['question'] + '\n' + 'إجابة: '
        

    #def doc_to_text(self, doc):
    #    return "{}\n هل هذه الجملة صحيحة أم خاطئة؟\nإجابة: ".format(
    #        doc["question"],
    #    )

    def doc_to_target(self, doc):
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, 'نعم')
        ll_false, _ = rf.loglikelihood(ctx, 'ال')
        return ll_true, ll_false

    def process_results(self, doc, results, save_all=False):
        pred = np.argmax(results)
        gold = {'صح': 0, 'خطأ': 1}[doc["answer"]]
        result = {
                "acc": pred == gold,
        }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

