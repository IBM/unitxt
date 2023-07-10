<div align="center">
    <img src="./assets/banner.png" alt="Image Description" width="100%" />
</div>
Unitxt is a python library for getting data fired up and set for utilization.
In one line of code, it preps a dataset or mixtures-of-datasets into an input-output format for training and evaluation.
We aspire to be simple, adaptable and transperant. 

Unitxt builds on separation. Separation allows adding a dataset, without knowing anything about the models using it. Separation allows training without caring for preprocessing, switching models without loading the data differently and changing formats (instruction\ICL\etc.) without changing anything else. 

# Installation 🦄


Install with git clone:

```bash
  git clone git@github.ibm.com:IBM-Research-AI/unitxt.git
  pip install ./unitxt
```

Install from GitHub with pip (in the future.)

```bash
  pip install git+ssh://git@github.ibm.com/IBM-Research-AI/unitxt.git
```

# Usage 🦄

## Load a dataset

```python
import unitxt

dataset = unitxt.load_dataset(['glue', 'stsb'])
```
Output in `dataset['train'][0]`:
```json
{
    "source": "Text Simalrity Regression; Sentence1: Some men are playing guitars., Sentence2: Three men are playing guitars and singing together.",
    "target": "3.20",
    "references": ["3.200000047683716"],
    "metrics": ["pearson", "spearman"],
    "task": "text-similarity-regression",
    "origin": "stsb"
}
```
## Load a dataset with different format or instructions

TBD

## Load a mixture of datasets

```python
unitext.load_dataset('glue', source='mixture')
```

# Why Unitxt? 🦄

Unitxt is construct in the light of the principles: (1) Simplicity (2) Adpatability (3) Transperancy
### 🦄 Simplicity
Everything is unitxt is simple and designed to feel natural and self explenatory.
### 🦄 Adaptability
Adding new datasets, loading recpepies, instructions and formattors is possible and encoureged!
### 🦄 Transperancy
The reosurces and formators of Unitxt are stored as shared datasets and therfore can easily reviewed by the crowed. Moreover, when assembling dataset with Unitxt it is very clear to others whats in it. 

# Alternatives

`PromptSource`: Static collection of datasets with static predfined formats and instructions.

`TaskSource`: Allows access to datasets with static predfined formats and instructions only for classification tasks.

`NaturalInstructions`: Static collection of datasets with instructions in one format.

`seqio`/ `FLAN-Collection`: Allows access to many datasets but is not easily adaptable to new formats and instructions.

`openai/evals`: ...

# Feature road map

- [x] Support for loading datasets with different formats
- [x] Support automatic evaluation with different metrics
- [x] Support for loading datasets with different instructions
- [x] Support for loading mixtures of datasets
- [x] support for few shot learning
- [x] support for deterministic loading
- [x] support for inference
- [ ] support for test only datasets
- [ ] 1000 datasets out of the box
- [ ] 10k datasets out of the box

# Comparision table
In the following table we compare features of different libraries for loading datasets in the following aspects: (1) allows different formats (2) allows different dataset loading instructions (3) allows loading mixtures of datasets (4) no code. 

| Library | Allows different formats | Allows loading with different instructions | Mixing datasets | mixing tasks | No code dataset addition | # supported tasks | few shot | detrminstic | inference support | training support | metric |
| --- | --- | --- | --- | --- |  --- |   --- |    --- |    --- |    --- |       --- |    --- | 
| Unitxt | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | TBD |


# Supported Tasks Out of the Box (N)

# Supported datasets Out of The Box (N)

