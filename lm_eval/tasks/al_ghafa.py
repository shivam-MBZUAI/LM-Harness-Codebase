"""
Boolq:

"""
import sys
import traceback

from lm_eval.base import MultipleChoiceTask
import os.path as osp
from datasets import load_dataset
from lm_eval.utils import ARA_DATA_DIR
from .sciq import SciQ
from .toxigen import ToxiGen
from .hellaswag import HellaSwag

alghafa_native_dataset_path = "OALL/AlGhafa-Arabic-LLM-Benchmark-Native"
alghafa_mt_dataset_path = "OALL/AlGhafa-Arabic-LLM-Benchmark-Translated"

al_ghafa_native_sub_datasets = ['mcq_exams_test_ar', 'meta_ar_dialects', 'meta_ar_msa', 'multiple_choice_facts_truefalse_balanced_task',
                'multiple_choice_grounded_statement_soqal_task', 'multiple_choice_grounded_statement_xglue_mlqa_task',
                'multiple_choice_rating_sentiment_no_neutral_task', 'multiple_choice_rating_sentiment_task',
                'multiple_choice_sentiment_task']

al_ghafa_mt_sub_datasets = ['arc_challenge_okapi_ar', 'arc_easy_ar', 'boolq_ar', 'copa_ext_ar', 'hellaswag_okapi_ar', 'sciq_ar', 'toxigen_ar',
   'mmlu_okapi_ar', 'openbook_qa_ext_ar', 'piqa_ar', 'race_ar']

def create_all_alghafa_native_tasks():
    task_dict = {f"alghafa_native-{sub}": create_task(alghafa_native_dataset_path,sub) for sub in al_ghafa_native_sub_datasets}
    return task_dict

def create_all_alghafa_mt_tasks():
    task_dict = {f"alghafa_mt-{sub}": create_task(alghafa_mt_dataset_path,sub) for sub in al_ghafa_mt_sub_datasets}
    return task_dict

def create_task(dataset_path,sub_dataset):
    if sub_dataset == 'sciq_ar':
        class SCIQ_AR(SciQ):
            VERSION = 0
            DATASET_PATH = dataset_path
            DATASET_NAME = sub_dataset
        return SCIQ_AR

    if sub_dataset == 'toxigen_ar':
        class TOXIGEN_AR(ToxiGen):
            VERSION = 0
            DATASET_PATH = dataset_path
            DATASET_NAME = sub_dataset
        return TOXIGEN_AR

    if sub_dataset == 'hellaswag_okapi_ar':
        class HELLASWAG_AR(HellaSwag):
            VERSION = 0
            DATASET_PATH = dataset_path
            DATASET_NAME = sub_dataset
        return HELLASWAG_AR

    class ACVA(AlGhafa_base):
        def __init__(self):
            super().__init__(dataset_path,sub_dataset)

    return ACVA

class AlGhafa_base(MultipleChoiceTask):
    VERSION = 0

    def __init__(self,dataset_path,dataset_name):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.download(dataset_path,dataset_name)

    def download(self,dataset_path,dataset_name):
        self.dataset = load_dataset(dataset_path, dataset_name)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _process_doc(self, doc):
        if self.dataset_name == 'boolq_ar':
            out_doc = {
                "query": doc["passage"] + '\n\nQuestion: ' + doc['question'],
                "choices": ["False", "True"],
                "gold": {False: 0, True: 1}[doc["answer"]],
            }
            return out_doc
        try:
            choices = []
            for i in range(1,10):
                k = f'sol{i}'
                if doc.get(k,None):
                    choices.append(doc[k])
            if doc.get('query',None):
                query = doc['query']
            else:
                query = doc['question']

            try:
                answer = doc["answer"]
            except:
                answer = doc["label"]

            out_doc = {
                "query": query,
                "choices": choices,
                "gold": answer,
            }
        except Exception as e:
            print("Error while processing doc: \n\n")
            print(f"Dataset: {self.dataset_path}-{self.dataset_name}")
            print(doc)
            print("\n\n")
            print(e)
            print(traceback.format_exc())
            sys.exit()
        return out_doc

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["query"] + "\nAnswer: "

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]





