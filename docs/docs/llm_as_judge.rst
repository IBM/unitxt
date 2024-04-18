.. _llm_as_judge:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
LLM As a Judge Metrics✨
=====================================

In this section you learn how to use LLM as judge metric by unitxt. LLM as a judge is a method for evaluation the
performance of a model based on the output of another model.

Using LLM  As a Judge in unitxt
----------------------------
Using LLM as a judge is extremely simple in unitxt. You should simply choose llm as a judge metric, and unitxt will do the rest...

Unitxt catalog includes a collection of preexisting LLM as judges that can be used like any other
metric.

To specify an LLM as judge metric, you can specify it in the dataset or in the recipe. For example:

.. code-block:: python

    card=cards.almost_evil,template=templates.qa.open.simple,metrics=[metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template]",



Adding new LLM As a Judge metric:
----------------------------

For a classical code-based metric (like F1, Rouge), the general evaluation flow is the following:
    1. load the dataset using a unitxt recipe (e.g. "cards.sst2")

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

In order to create new LLM as a judge metric, one should decide which model should be the judge, and
how to create it input text based on the prediction of the tested model.

Lets review an example of adding a LLM by judge metric:

.. code-block:: python

    import evaluate
    from datasets import load_dataset
    from unitxt.inference import HFPipelineBasedInferenceEngine

    # 1. Create the dataset
    dataset = load_dataset("unitxt/data", "card=cards.almost_evil,template=templates.qa.open.simple,"
                                          "metrics=[metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template]",
                           split='test')
    # 2. use inference module to infer based on the dataset inputs.
    inference_model = HFPipelineBasedInferenceEngine(model_name="google/flan-t5-small", max_new_tokens=32)
    predictions = inference_model.infer(dataset)
    # 3. create a metric and evaluate the results.
    metric = evaluate.load("unitxt/metric")
    scores = metric.compute(predictions=predictions, references=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]

In this case, we used the metric metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template, which uses flan t5
as a judge, and it use mt_bench recipe for creating the judging dataset.

In order to create new LLM as a judge metric, you should simply use the LLMAsJudge class. For example, lets see the definition
of metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template:


.. code-block:: python

    from unitxt import add_to_catalog
    from unitxt.inference import HFPipelineBasedInferenceEngine
    from unitxt.llm_as_judge import LLMAsJudge

    inference_model = HFPipelineBasedInferenceEngine(
        model_name="google/flan-t5-large", max_new_tokens=32
    )
    recipe = (
        "card=cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
        "template=templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
        "demos_pool_size=0,"
        "num_demos=0"
    )

    metric = LLMAsJudge(inference_model=inference_model, recipe=recipe)

    add_to_catalog(
        metric,
        "metrics.rag.model_response_assessment.llm_as_judge_by_flan_t5_large_on_hf_pipeline_using_mt_bench_template",
        overwrite=True,
    )

We can see, that each LLM as a judge metric needs two specifications:
    1. Inference engine with a model for judging (You can use any inference engine that implements InferenceEngine, and any desired model).

    2. Unitxt recipe for creating the judgment inputs.

Please note, that since the metric performs nested inference, there should be a consistency between the main recipe, and the judgment recipe.
    1. Since the judgment recipe uses the main recipe inputs and output, the names should match. In our example,
    card.almost_evil uses tasks.qa.open task, which specify the input field "question" and the output field "answers".
    On the other hand, cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template uses the task
    tasks.rag.model_response_assessment. This task defined as input the fields "question" - which is consistent
    with the main recipe field, and "model_output" - which is the standard name for the inference result. This task defines the
    output field "rating_label" - which is a standard name.

    2. Since LLM as a judge metric last step is extracting the judgment and passed it as a metric score, the template of the
    recipe should define postprocessor for the extraction. Since the unitxt scores are in scase of [0, 1], the postprocessor
    should convert the judgment to this scale. In our example, the card in the metric recipe -
    cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template, uses the template "templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template".
    This template specify for the judge how it expect the judgment format ("you must rate the response on a scale of 1
    to 10 by strictly following this format: "[[rating]]""), and on the other hand, it defines the processor for extracting
    the judgment. (postprocessors=[r"processors.extract_mt_bench_judgment"],). This processor simply extract the number within
    [[ ]] and divide it by 10 in order to scale to to [0, 1].