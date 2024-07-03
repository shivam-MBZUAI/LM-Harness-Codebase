"""
EXAMS: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering
https://www.aclweb.org/anthology/2020.emnlp-main.438/

EXAMS is a new benchmark dataset for cross-lingual and multilingual question answering for high school examinations.
It contains more than 24,000 high-quality high school exam questions in 26 languages, covering 8 language families and
24 school subjects from Natural Sciences and Social Sciences, among others. EXAMS offers a fine-grained evaluation
framework across multiple languages and subjects, which allows precise analysis and comparison of various models.

In this implementation we used only Arabic QA for evaluations. There are no Arabic QAs in training or devolpement.
We used 562 test Arabic QA for the evaluation.

Homepage: https://github.com/mhardalov/exams-qa/tree/main
"""
from lm_eval.base import MultipleChoiceTask
# from datasets import load_dataset
# import numpy as np
# import os.path as osp
# from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno, f1_score_multiclass
from lm_eval.utils import general_detokenize, camel_clean, PROMPT_DICT, ARA_DATA_DIR

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{hardalov-etal-2020-exams,
    title = "{EXAMS}: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering",
    author = "Hardalov, Momchil  and
        Mihaylov, Todor  and
        Zlatkova, Dimitrina  and
        Dinkov, Yoan  and
        Koychev, Ivan  and
        Nakov, Preslav",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.438",
    pages = "5427--5444",
    series = "EMNLP~'20"
}
"""


# TODO: Replace `NewTask` with the name of your Task.
class OALL_Exams_AR(MultipleChoiceTask):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "OALL/Arabic_EXAMS"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        que = camel_clean(doc["question"])
        out_doc = {
            "id": doc["id"],
            "subject": doc["subject"],
            "question": que,
            "query": "Question: " + que + "\nAnswer:",  # The query prompt.
            "choices": [camel_clean(doc[c]) for c in ["A", "B", "C", "D"]],  # The list of choices.
            "gold": ["A", "B", "C", "D"].index(doc["answer"]),  # The integer used to index into the correct element of `"choices"`.
        }
        return out_doc


    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]