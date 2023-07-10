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
  pip install unitxt
```

# Usage 🦄

## Load a dataset

```python
from datasets import load_dataset

dataset = load_dataset('unitxt/data', 'squad')
```

# Why Unitxt? 🦄

Unitxt is construct in the light of the principles: (1) Simplicity (2) Adpatability (3) Transperancy
### 🦄 Simplicity
Everything is unitxt is simple and designed to feel natural and self explenatory.
### 🦄 Adaptability
Adding new datasets, loading recpepies, instructions and formattors is possible and encoureged!
### 🦄 Transperancy
The reosurces and formators of Unitxt are stored as shared datasets and therfore can easily reviewed by the crowed. Moreover, when assembling dataset with Unitxt it is very clear to others whats in it. 

#

