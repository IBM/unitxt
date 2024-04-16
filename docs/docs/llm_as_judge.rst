.. _llm_as_judge:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
LLM As a Judge Metrics✨
=====================================

In this section you learn how to use LLM as judge metric by unitxt. LLM as a judge is a method for evaluation the
performance of a model based on the output of another model.

LLM  As a Judge Flow:
----------------------------

For a rule-based metric, the general evaluation flow is the following:
    1. create dataset

    2. use inference module to infer based on the dataset inputs.

    3. create a metric and evaluate the results.

In LLM as judge metric, we should feed a judge model by the predictions of the model we want to test, and ask it to judge
these prediction. The evaluation scores should be the predictions of the judge model.

Therefore, LLM as a judge flow:
    1. create dataset

    2. use inference module to infer based on the dataset inputs.

    3. create a metric and evaluate the results.
        3.1 create judging dataset, based on a desired specification (e.g. the desired template and format), and the prediction generated in (2.)

        3.2 getting a judge model, and infer it by the dataset generated in (3.1)

        3.3 extract the results from the judge predictions

Using LLM  As a Judge in unitxt
----------------------------

Using LLM as a judge is extremely simple in unitxt. You should simply choose llm as a judge metric, and unitxt will do the rest...

.. code-block:: python

    import evaluate
    from datasets import load_dataset
    from unitxt.inference import PipelineBasedInferenceEngine

    # 1. Create the dataset
    dataset = load_dataset("unitxt/data", "card=cards.almost_evil,template=templates.qa.open.simple,"
                                          "metrics=[metrics.rag.llm_as_judge.model_response_assessment.mt_bench_flan_t5]",
                           split='test')
    # 2. use inference module to infer based on the dataset inputs.
    inference_model = PipelineBasedInferenceEngine(model_name="google/flan-t5-small", max_new_tokens=32)
    predictions = inference_model.infer(dataset)
    # 3. create a metric and evaluate the results.
    metric = evaluate.load("unitxt/metric")
    scores = metric.compute(predictions=predictions, references=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]

In this case, we used the metric metrics.rag.llm_as_judge.model_response_assessment.mt_bench_flan_t5, which uses flan t5
as a judge, and it use mt_bench recipe for creating the judging dataset.

In order to create new LLM as a judge metric, you should simply use the LLMAsJudge class. For example, lets see the definition
of metrics.rag.llm_as_judge.model_response_assessment.mt_bench_flan_t5:


.. code-block:: python

    from unitxt import add_to_catalog
    from unitxt.inference import PipelineBasedInferenceEngine
    from unitxt.llm_as_judge import LLMAsJudge

    inference_model = PipelineBasedInferenceEngine(
        model_name="google/flan-t5-large", max_new_tokens=32
    )
    recipe = (
        "card=cards.llm_as_judge.model_response_assessment.mt_bench,"
        "template=templates.llm_as_judge.model_response_assessment.mt_bench,"
        "demos_pool_size=0,"
        "num_demos=0"
    )

    metric = LLMAsJudge(inference_model=inference_model, recipe=recipe)

    add_to_catalog(
        metric,
        "metrics.rag.llm_as_judge.model_response_assessment.mt_bench_flan_t5",
        overwrite=True,
    )

We can see, that each LLM as a judge metric needs two specifications:
    1. Inference engine with a model for judging (You can use any inference engine that implements InferenceEngine, and any desired model).

    2. Unitxt recipe for creating the judgment inputs.

Please note, that since the metric performs nested inference, there should be a consistency between the main recipe, and the judgment recipe.
    1. Since the judgment recipe uses the main recipe inputs and output, the names should match. In our example,
    card.almost_evil uses tasks.qa.open task, which specify the input field "question" and the output field "answers".
    On the other hand, cards.llm_as_judge.model_response_assessment.mt_bench uses the task
    tasks.llm_as_judge.rag.model_response_assessment. This task defined as input the fields "question" - which is consistent
    with the main recipe field, and "model_output" - which is the standard name for the inference result. This task defines the
    output field "rating_label" - which is a standard name.

    2. Since LLM as a judge metric last step is extracting the judgment and passed it as a metric score, the template of the
    recipe should define postprocessor for the extraction. Since the unitxt scores are in scase of [0, 1], the postprocessor
    should convert the judgment to this scale. In our example, the card in the metric recipe -
    cards.llm_as_judge.model_response_assessment.mt_bench, uses the template "templates.llm_as_judge.model_response_assessment.mt_bench".
    This template specify for the judge how it expect the judgment format ("you must rate the response on a scale of 1
    to 10 by strictly following this format: "[[rating]]""), and on the other hand, it defines the processor for extracting
    the judgment. (postprocessors=[r"processors.extract_mt_bench_judgment"],). This processor simply extract the number within
    [[ ]] and divide it by 10 in order to scale to to [0, 1].