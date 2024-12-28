# Analysing Bias in LLMs

### About

This project aims to analyse the biases present in LLMs with a focus on the Indian context. The project is divided into two tasks. The first task focuses on analysing the biases present in the language models by using the MLM task. The second task focuses on analysing the biases present in the language models when used in legal settings. The project uses the Hugging Face Transformers library for running the models. The project uses the dataset [nlp-fairness-for-india](https://github.com/google-research-datasets/nlp-fairness-for-india) base on the paper [Re-contextualizing Fairness in NLP: The Case of India](https://arxiv.org/abs/2209.12226).

### Presentation

A presentation based on the analysis made as a part of the project is available [here](https://www.canva.com/design/DAGaRiy5ShA/xvRmIe9U_qkCeeTiAJ5sIg/view?utm_content=DAGaRiy5ShA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h511ddf496c)

### Running Code
All the code is divided into notebooks. The code can be run by opening the notebooks in Google Colab or locally and running the cells. The code is written in Python and uses the Hugging Face Transformers library, and Pipelines for running the models.

The directory structure is descibed below which explains the structure of the code and the data files used, which will help in running the code. Additionally the notebooks also contain certain markdown cells which explain the code. 

### Dependencies
```
transformers
pandas
json
csv
plotly
```

### Directory Structure
```
.
├── README.md
├── task1 (this contains the code and data for task 1)
│   ├── mlm_analysis
│   │   ├── code
│   │   │   ├── code.ipynb
│   │   │   ├── display_results.ipynb
│   │   │   ├── get_results.ipynb
│   │   │   └── initial_analysis.ipynb
│   │   ├── data (the same structure is followed for all the folders in this directory and thus the structure of only one folder is described)
│   │   │   ├── adv_dec 
│   │   │   │   ├── final_results (all the final results for a given subfolder)
│   │   │   │   │   ├── non_stereotypes
│   │   │   │   │   │   ├── region_token_scores.json
│   │   │   │   │   │   ├── region_tokens.json (tokens that were predicted by the model but marked as non-stereotypes)
│   │   │   │   │   │   ├── religion_token_scores.json
│   │   │   │   │   │   └── religion_tokens.json
│   │   │   │   │   └── stereotypes
│   │   │   │   │       ├── region_results.json
│   │   │   │   │       ├── region_token_scores.json
│   │   │   │   │       ├── region_tokens.json (tokens that were predicted by the model but marked as stereotypes)
│   │   │   │   │       ├── religion_results.json
│   │   │   │   │       ├── religion_token_scores.json
│   │   │   │   │       └── religion_tokens.json
│   │   │   │   ├── prompts.json (original prompts)
│   │   │   │   ├── region_dataset.json (prompts where identity terms are replaced with region names)
│   │   │   │   ├── region_results.json (results for the region dataset - after MLM)
│   │   │   │   ├── religion_dataset.json (prompts where identity terms are replaced with religion names)
│   │   │   │   ├── religion_results.json (results for the religion dataset - after MLM)
│   │   │   │   └── results (intermediate results for the subfolder)
│   │   │   │       ├── region_results.json (combined results for the region dataset)
│   │   │   │       ├── religion_results.json (combined results for the religion dataset)
│   │   │   │       ├── scores_region_results.json 
│   │   │   │       └── scores_religion_results.json 
│   │   │   ├── adv_future
│   │   │   ├── adv_inc
│   │   │   ├── adv_past
│   │   │   ├── adv_perspective_shift
│   │   │   ├── adv_present
│   │   │   ├── negative_framing
│   │   │   ├── neutral_framing
│   │   │   ├── positive_framing
│   │   │   └── vanilla
│   │   └── results (for all the subfolders within the axes folder, results are present for the same set of subcategories and hence only one is described here)
│   │       ├── region 
│   │       │   ├── non_stereotype_csv (these are the tokens)
│   │       │   │   ├── framing.csv
│   │       │   │   ├── original.csv
│   │       │   │   ├── perspective.csv
│   │       │   │   ├── quantifier.csv
│   │       │   │   └── temporal.csv
│   │       │   ├── non_stereotype_scores_csv (these are cumulative scores for the tokens)
│   │       │   ├── non_stereotype_scores_plots (these are the plots for the cumulative scores)
│   │       │   ├── stereotype_csv
│   │       │   ├── stereotype_scores_csv
│   │       │   └── stereotype_scores_plots
│   │       └── religion
│   │           ├── non_stereotype_csv
│   │           │   ├── framing.csv
│   │           │   ├── original.csv
│   │           │   ├── perspective.csv
│   │           │   ├── quantifier.csv
│   │           │   └── temporal.csv
│   │           ├── non_stereotype_scores_csv
│   │           ├── non_stereotype_scores_plots
│   │           ├── stereotype_csv
│   │           ├── stereotype_scores_csv
│   │           └── stereotype_scores_plots
│   ├── not_complete_exp
│   │   ├── combined
│   │   │   ├── code.ipynb
│   │   │   ├── combined_dataset.json
│   │   │   ├── combined_results.json
│   │   │   ├── prompts.json
│   │   │   └── results
│   │   │       ├── combined_results.json
│   │   │       └── scores_combined_results.json
│   │   └── perplexity
│   │       ├── code.ipynb
│   │       ├── region_dataset.json
│   │       ├── region_results.json
│   │       ├── religion_dataset.json
│   │       └── religion_results.json
│   ├── notes.md (these are some notes I made for initial analysis, and to keep a track of the experiments I wanted to run)
│   ├── Results.md (these are some of the important results presented in a table format)
│   └── utils
│       ├── all_region_data.json
│       ├── all_religion_data.json
│       ├── region_annotated.csv
│       ├── region_tokens.txt
│       ├── religion_annotated.csv
│       └── religion_tokens.txt
└── task2 (this contains the code and data for task 2)
    ├── code
    │   ├── code.ipynb
    │   └── display_results.ipynb
    ├── data (original data files which are prsent for 10 models)
    │   ├── alpha.jsonl
    │   ├── beta.jsonl
    │   ├── . . . 
    ├── notes.md (these are some notes I made for initial analysis)
    ├── preprocessed_files
    │   ├── alpha.jsonl
    │   ├── beta.jsonl
    │   ├── . . .
    ├── results 
    │   ├── action_identity_results
    │   │   ├── alpha.jsonl
    │   │   ├── beta.jsonl
    │   │   ├── . . .
    │   ├── action_results
    │   │   ├── alpha.jsonl
    │   │   ├── beta.jsonl
    │   │   ├── . . .
    │   ├── combined_results
    │   │   ├── alpha.jsonl
    │   │   ├── beta.jsonl
    │   │   ├── . . .
    │   ├── gender_results
    │   │   ├── alpha.jsonl
    │   │   ├── beta.jsonl
    │   │   ├── . . .
    │   └── identity_term_results
    │       ├── alpha.jsonl
    │       ├── beta.jsonl
    │       ├── . . .
    └── utils
        ├── actions.txt
        └── identities.txt
```