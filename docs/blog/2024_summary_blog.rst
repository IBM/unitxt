.. title:: Unitxt 2024 End-of-Year Summary
:Authors: Unitxt Team
:Date: 2025-01-01

=================================================================================================
[01/01/2025] Unitxt 2024 End-of-Year Summary
=================================================================================================

**Authors**: Unitxt Team

``01/01/2025``

In 2024, Unitxt has evolved from its foundation as a specialized data preparation library into a comprehensive evaluation platform, driving significant impact within IBM.
This annual review highlights our key developments, achievements, and the strategic direction that has shaped our platform's growth.

From Vision to Reality: The Evolution of Unitxt
------------------------------------------------

Unitxt began with a clear mission to simplify data preparation for evaluation.
Its modular and flexible architecture, described in our `NAACL 2024 paper <https://aclanthology.org/2024.naacl-demo.21/>`_, proved valuable in data processing , leading to successful integrations with evaluation libraries including LM-Eval Harness Stanford's HELM `(Read HELM's blog post about our collaboration) <https://crfm.stanford.edu/2024/09/05/unitxt.html>`_.
These early achievements established a strong foundation for further development.

Is Unitxt a Data Preparation Library?
--------------------------------------

While Unitxt's origins lie in data preparation—where it brought structure and efficiency to evaluation workflows—its capabilities have expanded significantly.
One example of this complex combination of capabilities can be found in Unitxt's fully customizable `RAG evaluation tools <https://www.unitxt.ai/en/latest/docs/rag_support.html>`_ .
The integration with major evaluation libraries validated our approach to data processing and established the groundwork for broader applications.

Transitioning to an Evaluation Platform
----------------------------------------

The latter half of 2024 marked a strategic evolution in Unitxt's development.
As the LLM landscape expanded beyond text to encompass images, videos, tables, and other modalities, Unitxt's modular architecture enabled rapid adaptation to these new requirements.
This led to the development of comprehensive tools for evaluating models across multiple input types `(Read our blog post) <https://www.unitxt.ai/en/latest/blog/inference_engines_blog.html>`_.

Advancing LLM Evaluation
-------------------------

To address multimodal AI requirements, Unitxt now supports inference across diverse `external APIs and local inference libraries <https://www.unitxt.ai/en/latest/docs/inference.html>`_ . The platform's protocol has been enhanced to handle rich chat formats incorporating images, videos, and emerging modalities, positioning it for future technological advances `(Read more in our blog post about advanced multi-modal evaluation) <https://www.unitxt.ai/en/latest/blog/vision_robustness_blog.html>`_.

Setting New Standards with Benchmarks
----------------------------------------

The introduction of `Unitxt Benchmarks <https://www.unitxt.ai/en/latest/docs/benchmark.html>`_ represents a significant milestone, delivering a robust suite for constructing and analyzing comprehensive benchmarks. Full integration with IBM's `Blue Bench <https://www.unitxt.ai/en/latest/catalog/catalog.benchmarks.bluebench.html>`_ has streamlined complex evaluations into efficient operations, demonstrating our commitment to technical excellence and usability.
We call on the open-source community and all passionate researchers dedicated to robust and reproducible evaluation of AI systems to contribute their datasets and benchmarks to Unitxt!

Breaking Ground in Multimodal Understanding
--------------------------------------------

Unitxt has advanced the field through `new benchmarks in image understanding <https://www.unitxt.ai/en/latest/catalog/catalog.benchmarks.vision.html>`_ and, in collaboration with Stanford University's HELM library, developed the first `comprehensive table understanding benchmark <https://www.unitxt.ai/en/latest/catalog/catalog.benchmarks.tables_benchmark.html>`_.
This benchmark provides detailed analysis of LLM performance across various table structures, representations, and instructions—offering valuable insights for practical applications.

Advancing LLM Evaluation with AI Judges
----------------------------------------

The platform's robust infrastructure for accessing multiple inference APIs and libraries has enabled `sophisticated support for LLMs as judges <https://www.unitxt.ai/en/latest/docs/llm_as_judge.html>`_.
Moreover, Unitxt recently introduced a comprehensive set of `customizable LLM judges that evaluate based on user-defined criteria <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge_pairwise_user_criteria_no_catalog.py>`_.

Excellence in Data Preparation
-------------------------------

While expanding its capabilities, Unitxt maintains strong performance in data preparation for LLMs.
The platform's `data fusion tools <https://www.unitxt.ai/en/latest/unitxt.fusion.html>`_ have proven effective for creating large-scale, multi-dataset compilations for supervised fine-tuning.
Ongoing collaboration with IBM teams conducting large-scale supervised fine-tuning demonstrates the practical impact of these capabilities, with further developments planned for 2025.

Strategic Focus for 2025: Enhancing Accessibility
--------------------------------------------------

As Unitxt has matured into a sophisticated platform offering comprehensive capabilities in data preparation, inference, and quality estimation, accessibility becomes increasingly important. Building on 2024's improvements in error handling and documentation, the 2025 roadmap focuses on two key initiatives:
1.	Streamlining the Unitxt API for enhanced user interaction
2.	Developing intelligent navigation tools to optimize platform utilization
We are now testing internally an `AI assistant <https://github.com/IBM/unitxt/tree/main/src/unitxt/assistant>`_ that is specialized in Unitxt.
The Unitxt AI Assistant will provide our users real-time support, guiding them through complex workflows and offering contextual help based on their specific tasks.

The 2025 strategy aims to increase platform accessibility while maintaining the robust capabilities that drive value for our users.
This annual review reflects significant progress in Unitxt's development and implementation. We look forward to continued innovation and advancement in the year ahead.

