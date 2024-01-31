<div align="center">
    <img src="./assets/banner.png" alt="Image Description" width="100%" />
</div>

[![Button](https://img.shields.io/badge/Video-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Demo-pink?style=for-the-badge)](https://huggingface.co/spaces/unitxt/explore)
[![Button](https://img.shields.io/badge/Tutorial-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/docs/adding_dataset.html)
[![Button](https://img.shields.io/badge/Paper-pink?style=for-the-badge)](https://arxiv.org/abs/2401.14019)
[![Button](https://img.shields.io/badge/Documentation-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Catalog-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Contributers-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/PyPi-pink?style=for-the-badge)](https://pypi.org/project/unitxt/)


In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

 Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.

#
[![version](https://img.shields.io/pypi/v/unitxt)](https://pypi.org/project/unitxt/)
![license](https://img.shields.io/github/license/ibm/unitxt)
![python](https://img.shields.io/badge/python-3.8%20|%203.9-blue)
![tests](https://img.shields.io/github/actions/workflow/status/ibm/unitxt/tests.yml?branch=main&label=tests)
[![codecov](https://codecov.io/gh/IBM/unitxt/branch/main/graph/badge.svg?token=mlrWq9cwz3)](https://codecov.io/gh/IBM/unitxt)
![Read the Docs](https://img.shields.io/readthedocs/unitxt)
[![downloads](https://static.pepy.tech/personalized-badge/unitxt?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/unitxt)

#

https://github.com/IBM/unitxt/assets/23455264/baef9131-39d4-4164-90b2-05da52919fdf

### 🦄 Currently on Unitxt Catalog

![NLP Tasks](https://img.shields.io/badge/NLP_tasks-24-blue)
![Dataset Cards](https://img.shields.io/badge/Dataset_Cards-382-blue)
![Templates](https://img.shields.io/badge/Templates-186-blue)
![Formats](https://img.shields.io/badge/Formats-7-blue)
![Metrics](https://img.shields.io/badge/Metrics-49-blue)

### 🦄 Run Unitxt Exploration Dashboard

To launch unitxt graphical user interface run:
```
unitxt-explore
```

# 🦄 Contributors

Please install Unitxt from source by:
```
git clone git@github.com:IBM/unitxt.git
cd unitxt
pip install -e ".[dev]"
pre-commit install
```

# 🦄 Citation

If you use Unitxt in your research, please cite our paper:

```
@misc{unitxt,
      title={Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI},
      author={Elron Bandel and Yotam Perlitz and Elad Venezian and Roni Friedman-Melamed and Ofir Arviv and Matan Orbach and Shachar Don-Yehyia and Dafna Sheinwald and Ariel Gera and Leshem Choshen and Michal Shmueli-Scheuer and Yoav Katz},
      year={2024},
      eprint={2401.14019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

