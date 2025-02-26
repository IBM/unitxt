# ToRR Benchmark

** Official code repository for paper "ToRR: A Multifaceted Benchmark for Table Reasoning and Robustness" **

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

To use the **ToRR** benchmark, first ensure you have [`unitxt`](https://github.com/IBM/unitxt) installed. 

By default, ToRR Benchmark is registered in your local Unitxt catalog by the name "benchmarks.torr" when you install Unitxt([torr.py](torr.py) contains the main script for creating and registering the benchmark in Unitxt catalog). 

Refer to [this](https://www.unitxt.ai/en/latest/docs/benchmark.html#benchmarks) link for running the benchmark in your environment.


## Contact
For any help, kindly email us at: email-id(NAME)

## Citation

```
@article{
}
```