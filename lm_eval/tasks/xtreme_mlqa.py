"""
The Cross-lingual Natural Language Inference (XNLI) corpus is a crowd-sourced collection of 5,000 test and 2,500 dev pairs for the MultiNLI corpus.
The pairs are annotated with textual entailment and translated into 14 languages: French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese,
Thai, Chinese, Hindi, Swahili and Urdu. This results in 112.5k annotated pairs. Each premise can be associated with the corresponding hypothesis in the 15 languages,
summing up to more than 1.5M combinations. The corpus is made to evaluate how to perform inference in any language (including low-resources ones like Swahili or Urdu)
when only English NLI data is available at training time. One solution is cross-lingual sentence encoding, for which XNLI is an evaluation benchmark. The Cross-lingual
Ransfer Evaluation of Multilingual Encoders (XTREME) benchmark is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models.
It covers 40 typologically diverse languages (spanning 12 language families) and includes nine tasks that collectively require reasoning about different levels of syntax and semantics.
The languages in XTREME are selected to maximize language diversity, coverage in existing tasks, and availability of training data. Among these are many under-studied languages,
 such as the Dravidian languages Tamil (spoken in southern India, Sri Lanka, and Singapore), Telugu and Malayalam (spoken mainly in southern India), and the Niger-Congo languages Swahili and Yoruba, spoken in Africa.
"""

import os

from datasets import load_dataset

from lm_eval.base import rf, Task
from lm_eval.metrics import evaluate_response, mean
from lm_eval.utils import  camel_clean

_CITATION = """
"""


class XTREME_MLQA_AREN(Task):
    VERSION = 0

    def __init__(self):
        self.dataset = load_dataset("xtreme", "MLQA.ar.en")
        self.dataset['test'] = self.dataset['test'].select(range(500))

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _process_doc(self, doc):
        doc["context"] = camel_clean(doc["context"])
        doc["ref"] = camel_clean(doc['answers']['text'][0])
        doc["source"] = f"{doc['context']}\n{doc['question']}"
        return doc

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc['source']

    def doc_to_target(self, doc):
        return doc["ref"]

    def construct_requests(self, doc, ctx):
        try:
            continuation = rf.greedy_until(ctx, {"until": ["<|endoftext|>"]})
        except Exception as e:
            continuation = ""
            print("No output can be found from the llm.")
            print(e)
        return continuation

    def process_results(self, doc, results, save_all=False):
        completion = results[0].strip()
        score = evaluate_response(doc['source'], doc["ref"], completion)

        result = {
            "score": score,
        }

        if save_all:
            result["example"] = {
                "input": doc["source"],
                "ref": doc["ref"],
                "pred": completion,
                "score": score
            }

        return result

    def aggregation(self):
        return {
            "score": mean
        }

    def higher_is_better(self):
        return {
            "score": True
        }


class XTREME_MLQA_ENAR(XTREME_MLQA_AREN):
    VERSION = 0

    def __init__(self):
        self.dataset = load_dataset("xtreme", "MLQA.en.ar")
        self.dataset['test'] = self.dataset['test'].select(range(500))


    def _process_doc(self, doc):
        doc["question"] = camel_clean(doc["question"])
        doc["ref"] = doc['answers']['text'][0]
        doc["source"] = f"{doc['context']}\n{doc['question']}"
        return doc