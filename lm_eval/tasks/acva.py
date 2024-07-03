"""
Boolq: 

"""
from lm_eval.base import MultipleChoiceTask
import os.path as osp
from datasets import load_dataset
from lm_eval.utils import ARA_DATA_DIR



sub_datasets = ['Algeria', 'Ancient_Egypt', 'Arab_Empire', 'Arabic_Architecture', 'Arabic_Art', 'Arabic_Astronomy',
                'Arabic_Calligraphy', 'Arabic_Ceremony', 'Arabic_Clothing', 'Arabic_Culture', 'Arabic_Food',
                'Arabic_Funeral', 'Arabic_Geography', 'Arabic_History', 'Arabic_Language_Origin', 'Arabic_Literature',
                'Arabic_Math', 'Arabic_Medicine', 'Arabic_Music', 'Arabic_Ornament', 'Arabic_Philosophy',
                'Arabic_Physics_and_Chemistry', 'Arabic_Wedding', 'Bahrain', 'Comoros', 'Egypt_modern',
                'InfluenceFromAncientEgypt', 'InfluenceFromByzantium', 'InfluenceFromChina', 'InfluenceFromGreece',
                'InfluenceFromIslam', 'InfluenceFromPersia', 'InfluenceFromRome', 'Iraq', 'Islam_Education',
                'Islam_branches_and_schools', 'Islamic_law_system', 'Jordan', 'Kuwait', 'Lebanon', 'Libya',
                'Mauritania', 'Mesopotamia_civilization', 'Morocco', 'Oman', 'Palestine', 'Qatar', 'Saudi_Arabia',
                'Somalia', 'Sudan', 'Syria', 'Tunisia', 'United_Arab_Emirates', 'Yemen', 'communication',
                'computer_and_phone', 'daily_life', 'entertainment']


def create_all_tasks():
    task_dict = {f"acva-{sub}": create_task(sub) for sub in sub_datasets}
    return task_dict


def create_task(sub_dataset):
    class ACVA(ACVA_base):
        def __init__(self):
            super().__init__(sub_dataset)

    return ACVA

class ACVA_base(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "OALL/ACVA"
    DATASET_NAME = None
    
    def __init__(self,dataset_name):
        self.download(dataset_name)
    
    def download(self,dataset_name):
        self.dataset = load_dataset(ACVA_base.DATASET_PATH,dataset_name)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _process_doc(self, doc):
        out_doc = {
            "query": doc['question'],
            "choices": ["صح", "خطأ"],
            "gold": {"صح": 0, "خطأ": 1}[doc["answer"]],
        }
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


class ACVA_arab_empire(ACVA_base):
    def __init__(self,dataset_name = "Arab_Empire"):
        super().__init__(dataset_name)


