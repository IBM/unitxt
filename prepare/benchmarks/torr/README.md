# ToRR Benchmark

** Official code repository for paper "The Mighty ToRR: A Benchmark for Table Reasoning and Robustness" **

ToRR is an open-source benchmark designed by domain experts to evaluate model performance and robustness on table reasoning tasks.


## Overview

<div align="center">
    <img src="https://raw.githubusercontent.com/IBM/unitxt/main/assets/torr/overview.png" alt="Overview of ToRR Benchmark" width="100%" />
</div>

ToRR is designed to evaluate the table understanding and reasoning capabilities of state-of-the-art LLMs. 
With a strong focus on robustness, it examines how models respond to differing prompt configurations – table serialization formats, as well as table perturbations in a structured and repeatable manner.


It encompasses 10 datasets belonging to six diverse downstream tabular tasks from multiple domains.

<div align="center">
    <img src="https://raw.githubusercontent.com/IBM/unitxt/main/assets/torr/datasets.png" alt="The selected datasets for ToRR along with their properties. The 3 columns on the right reflect the required skills to solve each dataset." width="100%" />
</div>


## Usage

* Create conda environment
```
conda create -y -n torr_env python=3.10
conda activate torr_env
```

* Set environment variables

```
export UNITXT_ALLOW_UNVERIFIED_CODE=True
```

* Install Unitxt library

To work with the ToRR benchmark, install Unitxt in development mode

```
git clone https://github.com/IBM/unitxt.git
cd unitxt
pip install -e ".[dev]"
```

* Install additional dependencies as required

To run model inference on the ToRR benchmark, install required additional packages:

```bash
pip install -r prepare/benchmarks/torr/requirements.txt
```

* Sample code for running the benchmark

By default, ToRR Benchmark is registered in local Unitxt catalog by the name "benchmarks.torr" when you install Unitxt. Here is a sample code showing the benchmark use with an inference engine.

```
from unitxt import evaluate, load_dataset, settings
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)

test_dataset = load_dataset(
    "benchmarks.torr",
    split="test",
    use_cache=True,
)

# Infer using inference engine and LLM of your choice
model = HFPipelineBasedInferenceEngine(		# We used Together AI as inference engine
    model_name="google/flan-t5-base", max_new_tokens=512	
)

predictions = model(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print("Global scores:")
print(results.global_scores.summary)
print("Subsets scores:")
print(results.subsets.scores.summary)
```


## Additional Details

For details on ToRR benchmark integration in Unitxt, see the [registration script](torr.py) and for the internal configuration details of the benchmark refer to corresponding [Unitxt recipe file](../../../prepare/recipes/torr.py).


To customize ToRR benchmark, refer to [this example](../../../examples/evaluate_torr_custom_sizes.py).

For general details on Unitxt benchmark support, refer to 
[this](https://www.unitxt.ai/en/latest/docs/benchmark.html).


## Contact
For any help, kindly email us at: shir.ashury.tahan@ibm.com(Shir Ashury-Tahan)

## Citation

```
@article{ashury2025mighty,
  title={The Mighty ToRR: A Benchmark for Table Reasoning and Robustness},
  author={Ashury-Tahan, Shir and Mai, Yifan and C, Rajmohan and Gera, Ariel and Perlitz, Yotam and Yehudai, Asaf and Bandel, Elron and Choshen, Leshem and Shnarch, Eyal and Liang, Percy and Shmueli-Scheuer, Michal and others},
  journal={arXiv preprint arXiv:2502.19412},
  year={2025}
}
```