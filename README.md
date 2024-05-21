# CaseGNN & CaseGNN++
Code for CaseGNN (ECIR 2024 paper):

Title: [CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs](https://arxiv.org/abs/2312.11229)

Author: Yanran Tang, Ruihong Qiu, Yilun Liu, Xue Li and Zi Huang

And CaseGNN++ (Extension of CaseGNN):

Title: [CaseGNN++: Graph Contrastive Learning for Legal Case Retrieval with Graph Augmentation](https://arxiv.org/abs/2405.11791)

Author: Yanran Tang, Ruihong Qiu, Yilun Liu, Xue Li and Zi Huang

# Installation
Requirements can be seen in `/requirements.txt`

# Dataset
Datasets can be downloaded from [COLIEE2022](https://sites.ualberta.ca/~rabelo/COLIEE2022/) and [COLIEE2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/). 

Specifically, the downloaded COLIEE2022 folders `task1_train_files_2022` and `task1_test_files_2022` should be put into `/PromptCase/task1_train_2022/` and `/PromptCase/task1_test_2022/` respectively. 

The label file `task1_train_labels_2022.json` and `task1_test_labels_2022.json` shoule be put into folder `/label/`. 

COLIEE2022 folders should be set in a similar way. 

The final project file are as follows:

    ```
    $ ./CaseGNN/
    .
    ├── DATASET
    │   └── data_load.py
    ├── Grpah_generation
    │   ├── graph
    │   │   ├── graph_bin_2022
    │   │   └── graph_bin_2023
    │   └── TACG.py
    ├── Information_extraction  
    │   ├── coliee2022_ie    
    │   ├── coliee2023_ie
    │   ├── lexnlp             
    │   ├── stanford-openie
    │   ├── create_structured_csv.py
    │   ├── knowledge_graph.py
    │   └── relation_extractor.py             
    ├── label 
    │   ├── hard_neg_top50_train_2022.json
    │   ├── hard_neg_top50_train_2023.json
    │   ├── task1_test_labels_2022.json            
    │   ├── task1_test_labels_2023.json 
    │   ├── task1_train_labels_2022.json 
    │   ├── task1_train_labels_2023.json 
    │   ├── test_2022_candidate_with_yearfilter.json
    │   └── test_2023_candidate_with_yearfilter.json     
    ├── PromptCase
    │   ├── preprocessing
    │   │   ├── openaiAPI.py
    │   │   ├── process.py
    │   │   └── reference.py
    │   ├── promptcase_embedding
    │   ├── PromptCase_embedding_generation.py
    │   ├── task1_test_2022
    │   │   └── task1_test_files_2022
    │   ├── task1_test_2023
    │   │   └── task1_test_files_2023
    │   ├── task1_train_2022
    │   │   └── task1_train_files_2022
    │   └── task1_train_2023
    │       └── task1_train_files_2023
    ├── CaseGNN_2022.sh
    ├── CaseGNN_2023.sh
    ├── CaseGNN++_2022.sh
    ├── CaseGNN++_2023.sh
    ├── LegalFeatureExtraction.sh
    ├── RelationExtraction.sh
    ├── PromptcaseEmbeddingGeneration.sh
    ├── TACG.sh
    ├── main.py
    ├── model.py
    ├── train.py
    ├── main_casegnn2plus.py
    ├── model_casegnn2plus.py
    ├── train_casegnn2plus.py
    ├── EUGATConv.py
    ├── torch_metrics.py
    ├── requirements.txt
    └── README.md          
    ```

# Data Preparation
## 1. Information Extraction
- 1. Legal Feature Extraction

    - [PromptCase Preprocessing](https://github.com/yanran-tang/PromptCase?tab=readme-ov-file#preprocessing) is used to extracted the fact and issue from the cases. 

    - Run `. ./LegalFeatureExtraction.sh` to generate files in the following three folders:
        - `/PromptCase/task1_test_2022/processed/`, 
        - `/PromptCase/task1_test_2022/processed_new/`, which is the legal issues of cases, 
        - `/PromptCase/task1_test_2022/summary_test_2022_txt/`, which is the legal facts of cases. 
    
    - The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `LegalFeatureExtraction.sh`.


- 2. Relation Extraction
    - Run `. ./RelationExtraction.sh`.

    - The final relation triplets are in the folder `/Information_extraction/coliee2022_ie/coliee2022train(or test)_sum(or fact)/result/`.

    - The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `RelationExtraction.sh`.

    - The relation extraction is based on the [knowledge_graph_from_unstructured_text](https://github.com/varun196/knowledge_graph_from_unstructured_text) and [lexnlp](https://github.com/LexPredict/lexpredict-lexnlp/tree/master/lexnlp).

- Note: Legal feature extraction should be done first since the relation extraction is based on the extracted legal features.

- The extracted information can be also downloaded [here](https://drive.google.com/drive/folders/1Ck1KecF28xqsjDZK1fqVGF3BozmSsAb7?usp=sharing).


## 2. PromptCase Embedding Generation
- [PromptCase](https://github.com/yanran-tang/PromptCase/blob/main/PromptCase_model.py) is used to generate the case embedding (the feature of virtual global node)
    - Run `. ./PromptcaseEmbeddingGeneration.sh`. 
    - The generated case embedding and the according index list of cases are saved in folder `/PromptCase/promptcase_embedding/`
    - The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `PromptcaseEmbeddingGeneration.sh`.
- The generated PromptCase embedding can be also downloaded [here](https://drive.google.com/drive/folders/1TYc3RM6vbldQNmM5aNawdYy-tFS6IWbu?usp=sharing).


## 3. TACG Constrction
- TACG constrction utilises the result of Information Extraction and PromptCase Embedding, please ensure the folders of  `coliee2022_ie/coliee2022train(or test)_sum(or fact)/result/` and `/PromptCase/promptcase_embedding/` have been generated or downloaded.
- Run `. ./TACG.sh`
- The TACG graphs are saved in folder `/Graph_generation/graph/`

- The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `TACG.sh`.


# Model Training
## 1. CaseGNN Model Training
Run `. ./CaseGNN_2022.sh` and `. ./CaseGNN_2023.sh` for COLIEE2022 and COLIEE2023, respectively.

## 2. CaseGNN++ Model Training
Run `. ./CaseGNN++_2022.sh` and `. ./CaseGNN++_2023.sh` for COLIEE2022 and COLIEE2023, respectively.

Specifically, augmentation methods can be chosen to use for: 
- Positive samples only (--pos_aug)
- Random negative samples only (--ran_aug)
- Both positive and random negative samples (--pos_aug --ran_aug)

# Cite
If you find this repo useful, please cite
```
@article{CaseGNN++,
  author       = {Yanran Tang and
                  Ruihong Qiu and
                  Yilun Liu and
                  Xue Li and
                  Zi Huang},
  title        = {CaseGNN++: Graph Contrastive Learning for Legal Case Retrieval with Graph Augmentation},
  journal      = {CoRR},
  volume       = {abs/2405.11791},
  year         = {2024},
}

@inproceedings{CaseGNN,
  author       = {Yanran Tang and
                  Ruihong Qiu and
                  Yilun Liu and
                  Xue Li and
                  Zi Huang},
  title        = {CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed
                  Graphs},
  booktitle    = {ECIR},
  year         = {2024}
}

@inproceedings{PromptCase,
  author       = {Yanran Tang and
                  Ruihong Qiu and
                  Xue Li},
  title        = {Prompt-Based Effective Input Reformulation for Legal Case Retrieval},
  booktitle    = {ADC},
  year         = {2023}
}
```