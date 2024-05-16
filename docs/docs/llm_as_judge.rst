.. _llm_as_judge:

.. note::
   To follow this tutorial, ensure you have :ref:`unitxt installed <install_unitxt>`.

=====================================
LLM as a Judge Metrics Guide 📊
=====================================

Welcome to the LLM as a Judge Metrics Guide! This section will walk you through harnessing the power of LLM as judge (LLMaJ) metrics using the unitxt package. LLM as a judge provides a method to assess the performance of a model based on the judgments of another model.

In this guide, we'll explore three key aspects of LLMaJ:
    1. Utilizing LLM as judge as a metric in Unitxt
    2. Incorporating a new LLM as a Judge metric into Unitxt
    3. Assessing the quality of an LLM as a Judge metric

But first, let's start with an overview:

Overview
---------

An LLM as a Judge metric consists of several essential components:

1. The judge model, such as *Llama-3-8B-Instruct*, which evaluates the performance of other models.
2. The platform responsible for executing the judge model, such as Huggingface or OpenAI API.
3. The template used to construct prompts for the judge models. This template should be reflective of the judgment needs to be done and usually incorporate both the input and output of the evaluated model. For instance:

    .. code-block::
        "Please rate the clarity, coherence, and informativeness of the following summary on a scale of 1 to 10\\n Full text: {model_input}\\nSummary: {model_output}"

4. The format in which the judge model expects to receive prompts. For example:

    .. code-block::
        "<INST>{input}</INST>"

5. Optionally, a system prompt passed to the judge model. This can provide additional context for evaluation.

Understanding these components is crucial for effectively leveraging LLM as a Judge metrics in model evaluation and improvement processes. With this foundation, let's delve into how to utilize and create these metrics in the unitxt package.

Using LLM as a Judge in Unitxt
-------------------------------
Employing a pre-defined LLM as a judge metric is effortlessly achieved within unitxt.

The unitxt catalog boasts a variety of preexisting LLM as judges that seamlessly integrate into your workflow.

Let's delve into an example of evaluating a *flan-t5-small* model on the MT-Bench benchmark, specifically utilizing the single model rating variation. To accomplish this, we require the following:

1. A Unitxt dataset card containing MT-Bench inputs, which will serve as the input for our evaluated model.
2. A Unitxt template to be paired with the card. As the MT-Bench dataset already includes full prompts, there is no need to construct one using a template; hence, we'll opt for the *empty* template.
3. A unitxt format to be utilized with the card. Given that *flan* models do not demand special formatting of the inputs, we'll utilize the *empty* format here as well.
4. An LLMaJ metric leveraging the MT-Bench evaluation prompt.

Fortunately, all these components are readily available in the unitxt catalog, including a judge model based on *Mistral* from Huggingface that employs the MT-Bench format:

1. Dataset Card: *"cards.mt_bench.generation.english_single_turn"*.
2. Template: *"templates.empty"*.
3. Format: *"formats.empty"*.
4. LLMaJ Metric: *"metrics.llm_as_judge.rating.mistralai_Mistral_7B_Instruct_v0_2_huggingface_template_mt_bench_single_turn"*.

.. note::
   Pay attention!
   We are using the mistralai/Mistral-7B-Instruct-v0.2 model from Huggingface. This model requires you to agree to the terms to use it on the model page and set the HUGGINGFACE_TOKEN environment argument.

From here, constructing the full unitxt recipe string is standard and straightforward:
.. code-block:: text
    card=cards.mt_bench.generation.english_single_turn,
    template=templates.empty,
    format=formats.empty,
    metrics=[metrics.llm_as_judge.rating.mistralai_Mistral_7B_Instruct_v0_2_huggingface_template_mt_bench_single_turn]

Verifying Your Configuration
---------------------------
If you want to verify that your setup runs smoothly, follow the steps outlined above to ensure everything runs as expected.

.. code-block:: python
    import evaluate
    from datasets import load_dataset
    from unitxt.inference import HFPipelineBasedInferenceEngine
    from unitxt import evaluate

    # 1. Create the dataset
    card = "card=cards.mt_bench.generation.english_single_turn,"
    "template=templates.empty"
    "format=formats.empty,"
    "metrics=[metrics.llm_as_judge.rating.mistralai_Mistral_7B_Instruct_v0_2_huggingface_template_mt_bench_single_turn]"

    dataset = load_dataset("unitxt/data",
                            card,
                           split='test')
    # 2. use inference module to infer based on the dataset inputs.
    inference_model = HFPipelineBasedInferenceEngine(model_name="google/flan-t5-small", max_new_tokens=32)
    predictions = inference_model.infer(dataset)
    # 3. create a metric and evaluate the results.
    scores = metric.compute(predictions=predictions, references=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]



Creating a new LLM As a Judge Metric
-------------------------------------

To construct a new LLM as a Judge metric, several key components must be defined:

1. **Judge Model**: Select a model that will assess the performance of other models.
2. **Execution Platform**: Choose the platform responsible for executing the judge model, such as Huggingface or OpenAI API.
3. **Template**: Develop a template reflecting the criteria for judgment, incorporating both the input and output of the evaluated model.
4. **Format**: Specify the format in which the judge model expects to receive prompts.
5. **System Prompt (Optional)**: Optionally, include a system prompt to provide additional context for evaluation.
6. **Task**: Specify the `task` the judge should run. The task is mainly defined by the selected template.
More information will be provided in the example steps.

Let's walk through an example of creating a new LLM as a Judge metric, specifically recreating the MT-Bench judge metric (single model, rating variation:

1. **Select Judge Model**: We will utilize the *meta-llama/Meta-Llama-3-8B-Instruct* model from Huggingface as our judge model.
2. **Execution Platform**: We will opt to execute the model locally using Huggingface.

For this example, we'll create an `HFPipelineInferenceEngine` object with the model ID and generation parameters:

.. code-block:: python
    from unitxt.inference import HFPipelineInferenceEngine
    from unitxt.llm_as_judge import LLMAsJudge

    model_id = "meta-llama/llama-3-8b-instruct"
    inference_model = HFPipelineInferenceEngine(model_name=model_id, max_generated_tokens=256)

Note that if you wish to use a different platform for running your judge model, you can implement
a new `InferenceEngine` class and substitute it for `HFPipelineInferenceEngine`.
You can find the definition of the `InferenceEngine` abstract class and pre-built inference engines
(e.g., `OpenAiInferenceEngine`) in `src/unitxt/inference.py`.

3. **Define Template**: Design the template for the MT-Bench judge metric, typically comprising
instructions for the judge, input questions, and model responses. This template will define
how the judge evaluates the responses. Here's an example template definition:

.. code-block:: python
    from unitxt import add_to_catalog
    from unitxt.templates import InputOutputTemplate

    add_to_catalog(
        InputOutputTemplate(
            instruction="Please act as an impartial judge and evaluate the quality of the response provided"
            " by an AI assistant to the user question displayed below. Your evaluation should consider"
            " factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of"
            " detail of the response. Begin your evaluation by providing a short explanation. Be as"
            " objective as possible. After providing your explanation, you must rate the response"
            ' on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example:'
            ' "Rating: [[5]]".\n\n',
            input_format="[Question]\n{question}\n\n"
            "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
            output_format="[[{rating}]]",
            postprocessors=[
                r"processors.extract_mt_bench_rating_judgment",
            ],
        ),
        "templates.response_assessment.rating.mt_bench_single_turn",
        overwrite=True,
    )

Ensure the template includes a postprocessor for extracting the judgment from the judge model output and
passing it as a metric score. The postprocessor defines how the judge's rating is extracted from the output.
In our example, the template specify for the judge how it expect the judgment format ("you must rate the response on a scale of 1
to 10 by strictly following this format: "[[rating]]""), and on the other hand, it defines the processor for extracting
the judgment. (postprocessors=[r"processors.extract_mt_bench_rating_judgment"],). This processor simply extract the number within
[[ ]] and divide it by 10 in order to scale to to [0, 1].

4. **Define Format**: Define the format expected by the judge model for receiving prompts.
For Llama3 models, you can use the format already available in the Unitxt
catalog under *"formats.llama3_chat"*.

5. **Define System Prompt**: We will not use a system prompt in this example.

6. **Associate Template with Task**: Associate the template with a Unitxt task, specifying the inputs the
template expects to receive, such as *question*, *answer*, and *rating*. This allows the metric
to build the prompt to the judge model based on the template fields. For example, if the template
fields are *question*, *answer*, and *rating*, the predefined task named "rating.single_turn" can be used.

With these components defined, creating a new LLM as a Judge metric is straightforward:

.. code-block:: python

    from unitxt import add_to_catalog
    from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams)
    from unitxt.llm_as_judge import LLMAsJudge

    model_id = "meta-llama/llama-3-8b-instruct"
    gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=252)
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_id, parameters=gen_params
    )

    task = "rating.single_turn"
    format = "formats.llama3_chat"
    template = "templates.response_assessment.rating.mt_bench_single_turn"

    metric = LLMAsJudge(
        inference_model=inference_model,
        task=task,
        template=template,
        format=format,
        main_score="llama_3_8b_mt_bench_single_turn",
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.llama_3_8b_mt_bench_single_turn",
        overwrite=True,
    )




Note that the `LLMAsJudge` class can receive the boolean argument `strip_system_prompt_and_format_from_inputs`
(defaulting to True). When set to True, any system prompts or formatting in the inputs received by
the evaluated model will be stripped.