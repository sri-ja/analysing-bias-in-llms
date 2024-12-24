# Analysing Bias in LLMs

### Running Code
All the code is present in notebooks. The code can be run by opening the notebooks in Google Colab and running the cells. The code is written in Python and uses the Hugging Face Transformers library, and Pipelines for running the models.

### Dependencies
```
transformers
pipelines
pandas
json
csv
plotly
```

### Directory Structure
```
.
├── README.md
├── task1
│   ├── code.ipynb
│   ├── display_results.ipynb
│   ├── get_results.ipynb
│   ├── mlm_analysis
│   │   ├── adv_dec
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── adv_future
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── adv_inc
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── adv_past
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── adv_perspective_shift
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── adv_present
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── all_region_data.json
│   │   ├── all_religion_data.json
│   │   ├── combined
│   │   │   ├── code.ipynb
│   │   │   ├── combined_dataset.json
│   │   │   ├── combined_results.json
│   │   │   ├── prompts.json
│   │   │   └── results
│   │   │       ├── combined_results.json
│   │   │       └── scores_combined_results.json
│   │   ├── negative_framing
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── neutral_framing
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── positive_framing
│   │   │   ├── code.ipynb
│   │   │   ├── final_results
│   │   │   │   ├── region_results.json
│   │   │   │   ├── religion_results.json
│   │   │   │   ├── stereotype_tokens_region_results.json
│   │   │   │   └── stereotype_tokens_religion_results.json
│   │   │   ├── prompts.json
│   │   │   ├── region_dataset.json
│   │   │   ├── region_results.json
│   │   │   ├── religion_dataset.json
│   │   │   ├── religion_results.json
│   │   │   └── results
│   │   │       ├── region_results.json
│   │   │       ├── religion_results.json
│   │   │       ├── scores_region_results.json
│   │   │       └── scores_religion_results.json
│   │   ├── region_annotated.csv
│   │   ├── religion_annotated.csv
│   │   └── vanilla
│   │       ├── code.ipynb
│   │       ├── final_results
│   │       │   ├── region_results.json
│   │       │   ├── religion_results.json
│   │       │   ├── stereotype_tokens_region_results.json
│   │       │   └── stereotype_tokens_religion_results.json
│   │       ├── prompts.json
│   │       ├── region_dataset.json
│   │       ├── region_results.json
│   │       ├── religion_dataset.json
│   │       ├── religion_results.json
│   │       └── results
│   │           ├── region_results.json
│   │           ├── religion_results.json
│   │           ├── scores_region_results.json
│   │           └── scores_religion_results.json
│   ├── nlp-fairness-for-india
│   │   ├── caste_idterms.tsv
│   │   ├── datacard.pdf
│   │   ├── gender_idterms.tsv
│   │   ├── gender_proxy_idterms.tsv
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── region_annotations.tsv
│   │   ├── region_idterms.tsv
│   │   ├── region_individual_annotation.tsv
│   │   ├── religion_annotations.tsv
│   │   ├── religion_idterms.tsv
│   │   ├── religion_individual_annotation.tsv
│   │   └── templates.tsv
│   ├── notes.md
│   ├── perplexity
│   │   ├── code.ipynb
│   │   ├── region_dataset.json
│   │   ├── region_results.json
│   │   ├── region_tokens.txt
│   │   ├── religion_dataset.json
│   │   ├── religion_results.json
│   │   └── religion_tokens.txt
│   └── Results.md
├── task2
│   └── legal
│       ├── action_identity_results
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── action_results
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── actions.txt
│       ├── code.ipynb
│       ├── combined_results
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── display_results.ipynb
│       ├── gender_results
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── identities.txt
│       ├── identity_term_results
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── LLMs
│       │   ├── alpha.jsonl
│       │   ├── beta.jsonl
│       │   ├── delta.jsonl
│       │   ├── epsilon.jsonl
│       │   ├── eta.jsonl
│       │   ├── gamma.jsonl
│       │   ├── iota.jsonl
│       │   ├── theta.jsonl
│       │   └── zeta.jsonl
│       ├── notes.md
│       └── preprocessed_files
│           ├── alpha.jsonl
│           ├── beta.jsonl
│           ├── delta.jsonl
│           ├── epsilon.jsonl
│           ├── eta.jsonl
│           ├── gamma.jsonl
│           ├── iota.jsonl
│           ├── theta.jsonl
│           └── zeta.jsonl
```