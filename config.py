# Task name -> domain
TASK_CONFIG = {'Competition-Math': 'Academic', 'ProofWiki_Proof': 'Academic', 'ProofWiki_Reference': 'Academic',
               'Stacks_Proof': 'Academic', 'Stacks_Reference': 'Academic', 'Stein_Proof': 'Academic',
               'Stein_Reference': 'Academic', 'Trench_Proof': 'Academic', 'Trench_Reference': 'Academic',
               'TAD': 'Academic', 'TAS2': 'Academic', 'StackMathQA': 'Academic', 'APPS': 'Code',
               'CodeEditSearch': 'Code', 'CodeSearchNet': 'Code', 'Conala': 'Code', 'HumanEval-X': 'Code',
               'LeetCode': 'Code', 'MBPP': 'Code', 'RepoBench': 'Code', 'TLDR': 'Code', 'SWE-Bench-Lite': 'Code',
               'Apple': 'Finance', 'ConvFinQA': 'Finance', 'FinQA': 'Finance', 'FinanceBench': 'Finance',
               'HC3Finance': 'Finance', 'TAT-DQA': 'Finance', 'Trade-the-event': 'Finance', 'AY2': 'Web', 'ELI5': 'Web',
               'Fever': 'Web', 'TREx': 'Web', 'WnCw': 'Web', 'WnWi': 'Web', 'WoW': 'Web', 'zsRE': 'Web',
               'AILA2019-Case': 'Legal', 'AILA2019-Statutes': 'Legal', 'BSARD': 'Legal', 'BillSum': 'Legal',
               'CUAD': 'Legal', 'GerDaLIR': 'Legal', 'LeCaRDv2': 'Legal', 'LegalQuAD': 'Legal', 'REGIR-EU2UK': 'Legal',
               'REGIR-UK2EU': 'Legal', 'ArguAna': 'Web', 'CQADupStack': 'Web', 'FiQA': 'Finance', 'NFCorpus': 'Medical',
               'Quora': 'Web', 'SciDocs': 'Academic', 'SciFact': 'Academic', 'TopiOCQA': 'Web', 'Touche': 'Web',
               'Trec-Covid': 'Medical', 'ACORDAR': 'Web', 'CPCD': 'Web', 'ChroniclingAmericaQA': 'Web',
               'Monant': 'Medical', 'NTCIR': 'Web', 'PointRec': 'Web', 'ProCIS-Dialog': 'Web', 'ProCIS-Turn': 'Web',
               'QuanTemp': 'Web', 'WebTableSearch': 'Web', 'CARE': 'Medical', 'MISeD': 'Web', 'SParC': 'Web',
               'SParC-SQL': 'Web', 'Spider': 'Web', 'Spider-SQL': 'Web', 'LitSearch': 'Academic', 'CAsT_2019': 'Web',
               'CAsT_2020': 'Web', 'CAsT_2021': 'Web', 'CAsT_2022': 'Web', 'Core_2017': 'Web', 'Microblog_2011': 'Web',
               'Microblog_2012': 'Web', 'Microblog_2013': 'Web', 'Microblog_2014': 'Web',
               'PrecisionMedicine_2017': 'Medical', 'PrecisionMedicine_2018': 'Medical',
               'PrecisionMedicine_2019': 'Medical', 'PrecisionMedicine-Article_2019': 'Medical',
               'PrecisionMedicine-Article_2020': 'Medical', 'CliniDS_2014': 'Medical', 'CliniDS_2015': 'Medical',
               'CliniDS_2016': 'Medical', 'ClinicalTrials_2021': 'Medical', 'ClinicalTrials_2022': 'Medical',
               'ClinicalTrials_2023': 'Medical', 'DD_2015': 'Web', 'DD_2016': 'Web', 'DD_2017': 'Web',
               'FairRanking_2020': 'Academic', 'FairRanking_2021': 'Web', 'FairRanking_2022': 'Web',
               'Genomics-AdHoc_2004': 'Medical', 'Genomics-AdHoc_2005': 'Medical', 'Genomics-AdHoc_2006': 'Medical',
               'Genomics-AdHoc_2007': 'Medical', 'TREC-Legal_2011': 'Legal', 'NeuCLIR-Tech_2023': 'Web',
               'NeuCLIR_2022': 'Web', 'NeuCLIR_2023': 'Web', 'ProductSearch_2023': 'Web', 'ToT_2023': 'Web',
               'ToT_2024': 'Web', 'FoodAPI': 'Code', 'HuggingfaceAPI': 'Code', 'PytorchAPI': 'Code',
               'SpotifyAPI': 'Code', 'TMDB': 'Code', 'TensorAPI': 'Code', 'ToolBench': 'Code', 'WeatherAPI': 'Code',
               'ExcluIR': 'Web', 'Core17': 'Web', 'News21': 'Web', 'Robust04': 'Web', 'InstructIR': 'Web',
               'NevIR': 'Web', 'IFEval': 'Web'}

# Some tasks share the same document corpus
# corpus name -> tasks used this corpus
SHARE_CORPUS = {'AY2': ['AY2', 'ELI5', 'Fever', 'TREx', 'WnCw', 'WnWi', 'WoW', 'zsRE'],
                'ProCIS-Dialog': ['ProCIS-Dialog', 'ProCIS-Turn'],
                'CAsT_2019': ['CAsT_2019', 'CAsT_2020'],
                'CAsT_2021': ['CAsT_2021', 'CAsT_2022'],
                'PrecisionMedicine_2017': ['PrecisionMedicine_2017', 'PrecisionMedicine_2018'],
                'Article_2019': ['PrecisionMedicine-Article_2019', 'PrecisionMedicine-Article_2020'],
                'CliniDS_2014': ['CliniDS_2014', 'CliniDS_2015', 'CliniDS_2016'],
                'ClinicalTrials_2021': ['ClinicalTrials_2021', 'ClinicalTrials_2022'],
                'AdHoc_2004': ['Genomics-AdHoc_2004', 'Genomics-AdHoc_2005'],
                'AdHoc_2006': ['Genomics-AdHoc_2006', 'Genomics-AdHoc_2007'],
                'NeuCLIR_2022': ['NeuCLIR_2022', 'NeuCLIR_2023'],
                'ToT_2023': ['ToT_2023', 'ToT_2024']}

# get all tasks of a domain (Academic, Code, Web, Legal, Medical, Finance)
def get_tasks_by_domain(domain):
    assert domain in ['Academic', 'Code', 'Web', 'Legal', 'Medical', 'Finance']
    out = []
    for task in TASK_CONFIG:
        if TASK_CONFIG[task] == domain:
            out.append(task)

# return list of all tasks
def get_all_tasks():
    return list(TASK_CONFIG.keys())
