.. _llm_as_judge:

.. note::

    To follow this tutorial, ensure you have :ref:`unitxt installed <install_unitxt>`.

=====================================
LLM as a Judge Metrics Guide 📊
=====================================

This section will walk you through harnessing the power of LLM as judge (LLMaJ) metrics using the Unitxt package. LLM as a judge
provides a method to assess the performance of a model based on the judgments of another model.

In this guide, we'll explore three key aspects of LLMaJ:
    1. Utilizing LLM as judge as a metric in Unitxt.
    2. Incorporating a new LLM as a judge metric into Unitxt.
    3. Assessing the quality of an LLM as a judge metric.

But first, let's start with an overview:

Overview
---------

An LLM as a Judge metric consists of several essential components:

1. The judge model, such as *Llama-3-8B-Instruct* or *gpt-3.5-turbo*, which evaluates the performance of other models.
2. The platform responsible for executing the judge model, such as Huggingface or OpenAI API.
3. The template used to construct prompts for the judge model. This template should be reflective of the judgment needed and usually incorporates both the input and output of the evaluated model. For instance:

    .. code-block:: text

        Please rate the clarity, coherence, and informativeness of the following summary on a scale of 1 to 10\\n Full text: {model_input}\\nSummary: {model_output}

4. The format in which the judge model expects to receive prompts. For example:

    .. code-block:: text

        <INST>{input}</INST>

5. Optionally, a system prompt to pass to the judge model. This can provide additional context for evaluation.

Understanding these components is crucial for effectively leveraging LLM as a judge metrics. With this foundation, let's examine  how to utilize and create these metrics in the Unitxt package.

Using LLM as a Judge in Unitxt
-------------------------------
Employing a pre-defined LLM as a judge metric is effortlessly achieved within Unitxt.

The Unitxt catalog boasts a variety of preexisting LLM as judges that seamlessly integrate into your workflow.

Let's consider an example of evaluating a *flan-t5-small* model on the MT-Bench benchmark, specifically utilizing the single model rating evaluation part of the benchmark. In this part, we provide the LLM as a Judge, the input provided to the model and the output it generation. The LLM as Judge is asked to rate how well the output of the model address the request in the input.

To accomplish this evaluation, we require the following:

1. A Unitxt dataset card containing MT-Bench inputs, which will serve as the input for our evaluated model.
2. A Unitxt template to be paired with the card. As the MT-Bench dataset already includes full prompts, there is no need to construct one using a template; hence, we'll opt for the *empty* template, which just passes the input prompt from the dataset to the model.
3. A unitxt format to be utilized with the card. Given that *flan* models do not demand special formatting of the inputs, we'll utilize the *empty* format here as well.
4. An LLM as a judge metric leveraging the MT-Bench evaluation prompt.

Fortunately, all these components are readily available in the Unitxt catalog, including a judge model based on *Mistral* from Huggingface that employs the MT-Bench format.
From here, constructing the full unitxt recipe string is standard and straightforward:

.. code-block:: text

    card=cards.mt_bench.generation.english_single_turn,
    template=templates.empty,
    format=formats.empty,
    metrics=[metrics.llm_as_judge.rating.mistralai_Mistral_7B_Instruct_v0_2_huggingface_template_mt_bench_single_turn]

.. note::

   Pay attention!
   We are using the mistralai/Mistral-7B-Instruct-v0.2 model from Huggingface. Using this model requires you to agree to the Terms of Use on the model page and set the HUGGINGFACE_TOKEN environment argument. Other platforms might have different requirements. For example if you are using an LLM as judge based on the OpenAI platform, you will need to set your OpenAI api key.


The following code performs the desired evaluation:

.. code-block:: python

    from datasets import load_dataset
    from unitxt.inference import HFPipelineBasedInferenceEngine
    from unitxt import evaluate

    # 1. Create the dataset
    card = ("card=cards.mt_bench.generation.english_single_turn,"
            "template=templates.empty,"
            "format=formats.empty,"
            "metrics=[metrics.llm_as_judge.rating.mistral_7b_instruct_v0_2_huggingface_template_mt_bench_single_turn]"
            )

    dataset = load_dataset("unitxt/data",
                           card,
                           split='test')
    # 2. use inference module to infer based on the dataset inputs.
    inference_model = HFPipelineBasedInferenceEngine(model_name="google/flan-t5-small", max_new_tokens=32, use_fp16=True)
    predictions = inference_model.infer(dataset)

    # 3. create a metric and evaluate the results.
    scores = evaluate(predictions=predictions, data=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]



Creating a new LLM as a Judge Metric
-------------------------------------

To construct a new LLM as a Judge metric, several key components must be defined:

1. **Judge Model**: Select a model that will assess the performance of other models.
2. **Execution Platform**: Choose the platform responsible for executing the judge model, such as Huggingface or OpenAI API.
3. **The Judging Task**: This define the inputs the judge model expect to receive and its output. This is coupled with the template. Two common tasks are single model rating we saw above and pairwise model comparison, in which the outputs of two models is compared, to see which better addressed the required input.
4. **Template**: Develop a template reflecting the criteria for judgment, usually incorporating both the input and output of the evaluated model.
5. **Format**: Specify the format in which the judge model expects to receive prompts.
6. **System Prompt (Optional)**: Optionally, include a system prompt to provide additional context for evaluation.

Let's walk through an example of creating a new LLM as a Judge metric, specifically recreating the MT-Bench judge metric single-model-rating evaluation:

1. **Selecting a Judge Model**: We will utilize the *mistralai/Mistral-7B-Instruct-v0.2* model from Huggingface as our judge model.
2. **Selecting an Execution Platform**: We will opt to execute the model locally using Huggingface.

    For this example, we will use the *HFPipelineInferenceEngine* class:

    .. code-block:: python

        from unitxt.inference import HFPipelineInferenceEngine
        from unitxt.llm_as_judge import LLMAsJudge

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        inference_model = HFPipelineInferenceEngine(model_name=model_id, max_generated_tokens=256)


    .. note::

        If you wish to use a different platform for running your judge model, you can implement
        a new `InferenceEngine` class and substitute it with the `HFPipelineInferenceEngine`.
        You can find the definition of the `InferenceEngine` abstract class and pre-built inference engines
        (e.g., `OpenAiInferenceEngine`) in `src/unitxt/inference.py`.


3. **Selecting the Judging Task**: This is a standard Unitxt task that defines the api of the judge model. The task specifies the input fields expected by the judge model, such as "question" and "answer," in the example below, which are utilized in the subsequent template. Additionally, it defines the expected output field as a float type. Another significant field is "metrics," which is utilized for the (meta) evaluation of the judge, as explained in the following section. Currently supported tasks are "rating.single_turn" and "rating.single_turn_with_reference".

    .. code-block:: python

        from unitxt.blocks import Task
        from unitxt.catalog import add_to_catalog

        add_to_catalog(
            Task(
                inputs={"question": "str", "answer": "str"},
                outputs={"rating": "float"},
                metrics=["metrics.spearman"],
            ),
            "tasks.response_assessment.rating.single_turn",
            overwrite=True,
        )

4. **Define the Template**: We want to construct a template that is identical to the MT-Bench judge metric. Pay attention that this metric have field that are compatible with the task we chose ("question", "answer" and "rating").

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

    .. note::

        Ensure the template includes a postprocessor for extracting the judgment from the judge model output and
        passing it as a metric score. In our example, the template specifies for the judge the expected judgment format
        ("you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]""),
        and such, it also defines the processor for extracting the judgment. (postprocessors=[r"processors.extract_mt_bench_rating_judgment"],).
        This processor simply extract the number within [[ ]] and divide it by 10 in order to scale to to [0, 1].


5. **Define Format**: Define the format expected by the judge model for receiving prompts. For Mitral models, you can use the format already available in the Unitxt catalog under *"formats.models.mistral.instruction""*.

6. **Define System Prompt**: We will not use a system prompt in this example.

With these components defined, creating a new LLM as a Judge metric is straightforward:

.. code-block:: python

    from unitxt import add_to_catalog
    from unitxt.inference import HFPipelineBasedInferenceEngine
    from unitxt.llm_as_judge import LLMAsJudge

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    format = "formats.models.mistral.instruction"
    template = "templates.response_assessment.rating.mt_bench_single_turn"
    task = "rating.single_turn"

    inference_model = HFPipelineBasedInferenceEngine(
        model_name=model_id, max_new_tokens=256, use_fp16=True
    )
    model_label = model_id.split("/")[1].replace("-", "_").replace(".", "_").lower()
    model_label = f"{model_label}_huggingface"
    template_label = template.split(".")[-1]
    metric_label = f"{model_label}_template_{template_label}"
    metric = LLMAsJudge(
        inference_model=inference_model,
        template=template,
        task=task,
        format=format,
        main_score=metric_label,
    )

    add_to_catalog(
        metric,
        f"metrics.llm_as_judge.rating.{model_label}_template_{template_label}",
        overwrite=True,
    )



.. note::

    The *LLMAsJudge* class can receive the boolean argument *strip_system_prompt_and_format_from_inputs*
    (defaulting to *True*). When set to *True*, any system prompts or formatting in the inputs received by
    the evaluated model will be stripped.

Evaluating a LLMaJ metric (Meta-evaluation)
--------------------------------------------
But wait, we missed a step! We know the LLM as a judge we created worth anything?
The answer is: You evaluate it like any other model in Unitxt.
Remember the task we defined in the previous section?

    .. code-block:: python

        from unitxt.blocks import Task
        from unitxt.catalog import add_to_catalog

        add_to_catalog(
            Task(
                inputs={"question": "str", "answer": "str"},
                outputs={"rating": "float"},
                metrics=["metrics.spearman"],
            ),
            "tasks.response_assessment.rating.single_turn",
            overwrite=True,
        )

This task define the (meta) evaluation of our LLMaJ model.
We will fetch a dataset of MT-Bench inputs and models outputs, together with scores judged by GPT-4.
We will consider these GPT4 scores as our gold labels and evaluate our LLMaJ model by comparing its score on the model outputs
to the score of GPT4 using spearman correlation as defined in the task card.

We will create a card, as we do for every other Unitxt scenario:

.. code-block:: python

    from unitxt.blocks import (
        TaskCard,
    )
    from unitxt.catalog import add_to_catalog
    from unitxt.loaders import LoadHF
    from unitxt.operators import (
        CopyFields,
        FilterByCondition,
        RenameFields,
    )
    from unitxt.processors import LiteralEval
    from unitxt.splitters import RenameSplits
    from unitxt.test_utils.card import test_card

    card = TaskCard(
        loader=LoadHF(path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"),
        preprocess_steps=[
            RenameSplits({"train": "test"}),
            FilterByCondition(values={"turn": 1}, condition="eq"),
            FilterByCondition(values={"reference": "[]"}, condition="eq"),
            RenameFields(
                field_to_field={
                    "model_input": "question",
                    "score": "rating",
                    "category": "group",
                    "model_output": "answer",
                }
            ),
            LiteralEval("question", to_field="question"),
            CopyFields(field_to_field={"question/0": "question"}),
            LiteralEval("answer", to_field="answer"),
            CopyFields(field_to_field={"answer/0": "answer"}),
        ],
        task="tasks.response_assessment.rating.single_turn",
        templates=["templates.response_assessment.rating.mt_bench_single_turn"],
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card,
        "cards.mt_bench.response_assessment.rating.single_turn_gpt4_judgement",
        overwrite=True,
    )

This is a card for the first turn inputs of the MT-Bench benchmarks (without reference),
together with the outputs of multiple models to those inputs and the scores of GPT-4
to those outputs.

Now all we need to do is to load the card, with the template and format the judge model is expected to use,
and run it.

.. code-block:: python

    from datasets import load_dataset
    from unitxt.inference import HFPipelineBasedInferenceEngine
    from unitxt import evaluate

    # 1. Create the dataset
    card = ("card=cards.mt_bench.response_assessment.rating.single_turn_gpt4_judgement,"
            "template=templates.response_assessment.rating.mt_bench_single_turn,"
            "format=formats.models.mistral.instruction")

    dataset = load_dataset("unitxt/data",
                           card,
                           split='test')
    # 2. use inference module to infer based on the dataset inputs.
    inference_model = HFPipelineBasedInferenceEngine(model_name="mistralai/Mistral-7B-Instruct-v0.2",
                                                     max_new_tokens=256,
                                                     use_fp16=True)
    predictions = inference_model.infer(dataset)
    # 3. create a metric and evaluate the results.
    scores = evaluate(predictions=predictions, data=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]

The output of this code is:

.. code-block:: text

    ('spearmanr', 0.18328402960291354)
    ('score', 0.18328402960291354)
    ('score_name', 'spearmanr')
    ('score_ci_low', 0.14680574316651868)
    ('score_ci_high', 0.23030798909064645)
    ('spearmanr_ci_low', 0.14680574316651868)
    ('spearmanr_ci_high', 0.23030798909064645)

We can see the Spearman correlation is *0.18*, which is considered low.
This means *"mistralai/Mistral-7B-Instruct-v0.2"* is not a good model to act as an LLM as a Judge,
at least when using the MT-Bench template.

In order to understand precisely why it is so, examination of the outputs of the model is needed.
In this case, it seems Mistral is having difficulties outputting the scores in the double square brackets format.
An example for the model output is:

.. code-block:: text

    Rating: 9

    The assistant's response is engaging and provides a good balance between cultural experiences and must-see attractions in Hawaii. The description of the Polynesian Cultural Center and the Na Pali Coast are vivid and evoke a sense of wonder and excitement. The inclusion of traditional Hawaiian dishes adds depth and authenticity to the post. The response is also well-structured and easy to follow. However, the response could benefit from a few more specific details or anecdotes to make it even more engaging and memorable.