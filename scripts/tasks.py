# Format  Task_name: [--tasks, --num_fewshot, 'task_specific_args']
base_tasks_en = {
    # knowledge
    "mmlu": ["mmlu-*", 0, '-'],
    "race": ["race", 0, '-'],

    # commonsense reasoning
    "hellaswag": ["hellaswag", 0, '-'],
    "piqa": ["piqa", 0, '-'],
    "boolq": ["boolq", 0, '-'],
    "siqa": ["siqa", 0, "-"],
    "arc_challenge": ["arc_challenge", 0, '-'],
    "openbookqa": ["openbookqa", 0, '-'],
    "winogrande": ["winogrande", 0, '-'],

    # misinformation, bias
    "truthfulqa": ["truthfulqa_mc", 0, '-'],
    "crowspairs": ["crows_pairs_english_*", 0, '-'],
}

base_tasks_ar = {
    # knowledge
    "exams_ar": ["exams_ar", 0, '-'],
    "mmlu_hu_ar": ["mmlu_hu_ar*", 0, '-'],
    "mmlu_ar": ["mmlu_ar*", 0, '-'],
    "digitised_ar": ["digitised_ar", 0, '-'],

    # commonsense reasoning
    "hellaswag_ar": ["hellaswag_ar", 0, "-"],
    "piqa_ar": ["piqa_ar", 0, "-"],
    "boolq_ar": ["boolq_ar", 0, "-"],
    "siqa_ar": ["siqa_ar", 0, "-"],
    "arc_challenge_ar": ["arc_challenge_ar", 0, "-"],
    "openbookqa_ar": ["openbookqa_ar", 0, '-'],

    # "Misinformation, bias, toxicity
    "truthfulqa_mc_ar": ["truthfulqa_mc_ar", 0, '-'],
    "crowspairs_ar": ["crows_pairs_ar", 0, '-'],

    "agqa": ["agqa", 0, '-'],
    "agrc": ["agrc", 0, '-'],
}

base_tasks_hi = {
    "mmlu_hi": ["mmlu_hi", 0, '-'],
    "hellaswag_hi": ["hellaswag_hi", 0, "-"],
    "arc_hi": ["arc_hi", 0, "-"],
    "truthfulqa_hi": ["truthfulqa_hi", 0, '-'],
}

generation_gpt4_eval_tasks = {
    'vicuna': ["vicuna", 'generic', '-'],
    'seeds': ["seeds", 'generic', '-'],
    'iqeval': ["iqeval", "single-with-ref", "-"],
    'itc': ["itc", 'single-with-ref', '-'],
    'quant': ["quant", 'single-with-ref', '-'],
    # 'safety_gen':["safety",'safety','-'],
    # "summary": ["summary", 'summary', '-'],
    # 'safety_gen':["safety",'preference','-'],
    # 'vicuna':["vicuna",'preference','-'],
    # 'safety_gen': ["safety", 'helpful', '-'],
    # 'vicuna': ["vicuna", 'helpful', '-'],
}

open_llm_en_tasks = {
    # knowledge
    "mmlu": ["mmlu-*", 5, '-'],

    # commonsense reasoning
    "hellaswag": ["hellaswag", 10, '-'],
    "arc_challenge": ["arc_challenge", 25, '-'],
    "winogrande": ["winogrande", 5, '-'],

    # misinformation, bias
    "truthfulqa": ["truthfulqa_mc", 0, '-'],

    # maths
    # "gsm8k": ["gsm8k", 5, '-'],
}
open_llm_ar_tasks = {
    "mmlu_ar": ["mmlu_ar*", 5, '-'],

    # commonsense reasoning
    "hellaswag_ar": ["hellaswag_ar", 10, "-"],
    "arc_challenge_ar": ["arc_challenge_ar", 25, "-"],

    # misinformation, bias
    "truthfulqa": ["truthfulqa_mc", 0, '-'],
}

math_tasks = {
    # "lila": ['lila-*', 5, '-'],
    # 'math': ['math_*', 0, '-'],
    "gsm8k": ["gsm8k", 5, '-'],

    'math': [
        'minerva_math_algebra,'
        'minerva_math_counting_and_prob,'
        'minerva_math_geometry,'
        'minerva_math_intermediate_algebra,'
        'minerva_math_num_theory,'
        'minerva_math_prealgebra,'
        'minerva_math_precalc',0,'-'],

    'agieval_en': ['agieval_en', 0, '-'],

    "bbh": ["bbh_zeroshot_penguins_in_a_table,bbh_zeroshot_boolean_expressions,bbh_zeroshot_multistep_arithmetic_two",
            0, '-'],
    "drop": ["drop", 0, '-'],
    "gpqa_diamond_zeroshot": ["gpqa_diamond_zeroshot", 0, '-'],

    # "iq_test":["iq_test",0,'-'],

    "mmlu_stem": ["mmlu_stem", 5, '-'],
    "arc_challenge_math": ["arc_challenge", 25, '-'],
    "hellaswag_math": ["hellaswag", 10, '-'],
}

uae_tasks = {
    "uae_en": ["uae_en", 0, '-'],
    "uae_ar": ["uae_ar", 0, '-'],
    "gulf_dialect": ["gulf_dialect", 0, '-'],

    "acva":['acva-*',0,'-'],
    # "alghafa_mt": ["alghafa_mt-*",0,'-'],
    "alghafa_native": ["alghafa_native-*",0,'-'],

    'alghafa_mt-arc_challenge_okapi_ar': ['alghafa_mt-arc_challenge_okapi_ar', 0, '-'],
    'alghafa_mt-arc_easy_ar': ['alghafa_mt-arc_easy_ar', 0, '-'],
    'alghafa_mt-boolq_ar': ['alghafa_mt-boolq_ar', 0, '-'],
    'alghafa_mt-copa_ext_ar': ['alghafa_mt-copa_ext_ar', 0, '-'],
    'alghafa_mt-mmlu_okapi_ar': ['alghafa_mt-mmlu_okapi_ar', 0, '-'],
    'alghafa_mt-openbook_qa_ext_ar': ['alghafa_mt-openbook_qa_ext_ar', 0, '-'],
    'alghafa_mt-piqa_ar': ['alghafa_mt-piqa_ar', 0, '-'],
    'alghafa_mt-race_ar': ['alghafa_mt-race_ar', 0, '-'],
    "oall_exams_ar": ["oall_exams_ar",0,'--save_eval_examples'],
    "oall_mmlu_ar": ["oall_mmlu_ar*",0,"-"],
}

long_context_tasks = {
    "needle_in_haystack": ['needle_in_haystack', 0, '--save_eval_examples'],
    "quality": ['quality', 0, '-'],
    "tpo": ['tpo', 0, '-'],

    "needle_in_haystack_ar": ['needle_in_haystack_ar', 0, '--save_eval_examples'],
    "ardc_long_context": ["ardc_long_context", 0, '-'],
}

legacy_tasks = {
    # translation
    "iwslt17-en-ar": ['iwslt17-en-ar', 0, '--save_eval_examples'],
    "iwslt17-ar-en": ['iwslt17-ar-en', 0, '--save_eval_examples'],

    # safety
    "safety_helpful": ["hh-rlhf_safety,pku_safety,pku_safety_ar,hh-rlhf_helpful,shp_helpful,pku_helpful,pku_helpful_ar",
                       0, '-'],

    # cross lingual
    "xtreme_ar_en": ['xtreme_ar_en', 0, '--save_eval_examples'],
    "xtreme_en_ar": ['xtreme_en_ar', 0, '--save_eval_examples'],
}


armmlu_tasks = {
    'ArabicMMLU': ['ArabicMMLU']
}