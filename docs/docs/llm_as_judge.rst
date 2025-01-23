.. _llm_as_judge:

.. note::

    To follow this tutorial, ensure you have :ref:`unitxt installed <install_unitxt>`.

=====================================
LLM as a Judge Metrics Guide 📊
=====================================

This section will walk you through harnessing the power of LLM as judge (LLMaJ) metrics using the Unitxt package. LLM as a judge
provides a method to assess the performance of a model based on the judgments of another model.

When to use LLM as Judge
------------------------

LLMs as judges are most useful when
    1. You don't have ground truth (references) to compare with
    2. When you have ground truth, but comparing the ground truth to the model response is non-trivial (e.g. requires semantic understanding)
    3. When you want to assess specific properties of the model's output that can easily expressed via an LLM prompt (e.g. does the model response contain profanity).

Disadvantages of LLM as Judge
-----------------------------

While LLMs as Judges are powerful and effective in many cases, they have some drawbacks:
    1. Good LLM as Judges are often large models with relatively high inference latency.
    2. Deploying large LLMs is difficult and may require API access to external services.
    3. Not all LLMs (including large ones) can serve as good judges - their assessment may not correlate with human judgements and can also be biased.
       This means that unless you have a prior indication that the LLM you use is a good judge for your task, you need to evaluate its judgements and see they match your expectations.


Using LLMs
-----------
In this guide, we'll explore three key aspects of LLMaJ:
    1. Utilizing LLM as judge as a metric in Unitxt.
    2. Incorporating a new LLM as a judge metric into Unitxt.
    3. Assessing the quality of an LLM as a judge metric.

But first, let's start with an overview:

Overview
---------

An LLM as a Judge metric consists of several essential components:

1. The judge model, such as *Llama-3-8B-Instruct* or *gpt-3.5-turbo*, which evaluates the performance of other models.
2. The platform responsible for executing the judge model, such as Huggingface, OpenAI API and IBM's deployment platforms such as WatsonX and RITS.
   A lot of these model and catalog combinations are already predefined in our catalog. The models are prefixed by metrics.llm_as_judge.direct followed by the platform and the model name.
   For instance, metrics.llm_as_judge.direct.rits.llama3_1_70b refers to llama3 70B model that uses RITS deployment service.

3. The criteria to evaluate the model's response. There are predefined criteria in the catalog and the user can also define a custom criteria.
   Each criteria specifies fine-grained options that help steer the model to evaluate the response more precisely.
   For instance the critertion "metrics.llm_as_judge.direct.criterias.answer_relevance" quantifies how much the model's response is relevant to the user's question.
   It has four options that the model can choose from and they are excellent, acceptable, could be improved and bad. Each option also has a description of itself and a score associated with it.
   The model uses these descriptions to identify which option the given response is closest to and returns them.
   The user can also specify their own custom criteria. An example of this is included under the section **Creating a custom criteria**.
   The user can specify more than one criteria too. This is illustrated in the **End to end example** section
4. The Context fields are the additional fields beyond the evaluated response that are passed to the LLM as judge. This could be the reference answer, the question or the context provided to the model etc.
    In the example below, the question that was input to the model is passed as a context field.

Understanding these components is crucial for effectively leveraging LLM as a judge metrics. With this foundation, let's examine  how to utilize and create these metrics in the Unitxt package.

Using LLM as a Judge in Unitxt
-------------------------------
Employing a pre-defined LLM as a judge metric is effortlessly achieved within Unitxt.

The Unitxt catalog boasts a variety of preexisting LLM as judges that seamlessly integrate into your workflow.

Let's consider an example of evaluating a model's responses for relevance to the questions.

To accomplish this evaluation, we require the following:

1. The questions that were input to the model
2. The judge model and its deployment platform
3. The pre-defined criteria, which in this case is metrics.llm_as_judge.direct.criterias.answer_relevance.

We pass the criteria to the judge model's metric as criteria and the question as the context fields.

.. code-block:: python

   data = [
    {"question": "Who is Harry Potter?"},
    {"question": "How can I protect myself from the wind while walking outside?"},
    {"question": "What is a good low cost of living city in the US?"},
    ]

    criteria = "metrics.llm_as_judge.direct.criterias.answer_relevance"
    metrics = [
    f"metrics.llm_as_judge.direct.rits.llama3_1_70b[criteria={criteria}, context_fields=[question]]"
    ]

    dataset = create_dataset(
        task="tasks.qa.open", test_set=data, metrics=metrics, split="test"
    )

Once the metric is created, a dataset is created for the appropriate task.

.. code-block:: python

    dataset = create_dataset(task="tasks.qa.open", test_set=data, metrics=metrics, split="test")

The model's responses are then evaluated by the judge model as follows:

.. code-block:: python

    predictions = [
        """Harry Potter is a young wizard who becomes famous for surviving an attack by the dark wizard Voldemort, and later embarks on a journey to defeat him and uncover the truth about his past.""",
        """You can protect yourself from the wind by wearing windproof clothing, layering up, and using accessories like hats, scarves, and gloves to cover exposed skin.""",
        """A good low-cost-of-living city in the U.S. is San Francisco, California, known for its affordable housing and budget-friendly lifestyle.""",
    ]

    results = evaluate(predictions=predictions, data=dataset)

    print("Global Scores:")
    print(results.global_scores.summary)

    print("Instance Scores:")
    print(results.instance_scores.summary)


Positional Bias
--------------------------------------------
Positional bias determines if the judge model favors an option owing to its placement within the list of available options rather than its intrinsic merit.
Unitxt reports if the judge model has positional bias in the instance level summary.

Creating a custom criteria
-------------------------------------
As described above, the user can either choose a pre-defined criteria from the catalog or define their own criteria. Below is an example of how the user can define their own criteria.
The criteria must have options and their descriptions for the judge model to choose from.
Below is an example where the user mandates that the model respond with the temperature in both Celsius and Fahrenheit. The various possibilities are described in the options and each option is associated with a score that is specified in the score map.

.. code-block:: python

    from unitxt.llm_as_judge_constants import  CriteriaWithOptions

    criteria = CriteriaWithOptions.from_obj(
        {
            "name": "Temperature in Fahrenheit and Celsius",
            "description": "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?",
            "options": [
                {
                    "name": "Correct",
                    "description": "The temperature reading is provided in both Fahrenheit and Celsius.",
                },
                {
                    "name": "Partially Correct",
                    "description": "The temperature reading is provided either in Fahrenheit or Celsius, but not both.",
                },
                {
                    "name": "Incorrect",
                    "description": "There is no numerical temperature reading in the response.",
                },
            ],
            "option_map": {"Correct": 1.0, "Partially Correct": 0.5, "Incorrect": 0.0},
        }
    )


End to end example
--------------------------------------------
Unitxt can also obtain model's responses for a given dataset and then run LLM-as-a-judge evaluations on the model's responses.
Here, we will get llama-3.2 1B instruct's responses and then evaluate them for answer relevance, coherence and conciseness using llama3_1_70b judge model

.. code-block:: python

    criterias = ["answer_relevance", "coherence", "conciseness"]
    metrics = [
    "metrics.llm_as_judge.direct.rits.llama3_1_70b"
    "[context_fields=[context,question],"
    f"criteria=metrics.llm_as_judge.direct.criterias.{criteria},"
    f"score_prefix={criteria}_]"
    for criteria in criterias
    ]
    dataset = load_dataset(
        card="cards.squad",
        metrics=metrics,
        loader_limit=10,
        max_test_instances=10,
        split="test",
    )

We use CrossProviderInferenceEngine for inference.

.. code-block:: python

    inference_model = CrossProviderInferenceEngine(
        model="llama-3-2-1b-instruct", provider="watsonx"
    )

    predictions = inference_model.infer(dataset)

    gold_answers = [d[0] for d in dataset["references"]]

    # Evaluate the predictions using the defined metric.
    evaluated_predictions = evaluate(predictions=predictions, data=dataset)
    evaluated_gold_answers = evaluate(predictions=gold_answers, data=dataset)

    print_dict(
        evaluated_predictions[0],
        keys_to_print=[
            "source",
            "score",
        ],
    )
    print_dict(
        evaluated_gold_answers[0],
        keys_to_print=[
            "source",
            "score",
        ],
    )

    for criteria in criterias:
        logger.info(f"Scores for criteria '{criteria}'")
        gold_answer_scores = [
            instance["score"]["instance"][f"{criteria}_llm_as_a_judge_score"]
            for instance in evaluated_gold_answers
        ]
        gold_answer_position_bias = [
            int(instance["score"]["instance"][f"{criteria}_positional_bias"])
            for instance in evaluated_gold_answers
        ]
        prediction_scores = [
            instance["score"]["instance"][f"{criteria}_llm_as_a_judge_score"]
            for instance in evaluated_predictions
        ]
        prediction_position_bias = [
            int(instance["score"]["instance"][f"{criteria}_positional_bias"])
            for instance in evaluated_predictions
        ]

        logger.info(
            f"Scores of gold answers: {statistics.mean(gold_answer_scores)} +/- {statistics.stdev(gold_answer_scores)}"
        )
        logger.info(
            f"Scores of predicted answers: {statistics.mean(prediction_scores)} +/- {statistics.stdev(prediction_scores)}"
        )
        logger.info(
            f"Positional bias occurrence on gold answers: {statistics.mean(gold_answer_position_bias)}"
        )
        logger.info(
            f"Positional bias occurrence on predicted answers: {statistics.mean(prediction_position_bias)}\n"
        )

.. code-block:: text

    Output with 100 examples

    Scores for criteria 'answer_relevance'
    Scores of gold answers: 0.9625 +/- 0.14811526360619054
    Scores of predicted answers: 0.5125 +/- 0.4638102516061385
    Positional bias occurrence on gold answers: 0.03
    Positional bias occurrence on predicted answers: 0.12

    Scores for criteria 'coherence'
    Scores of gold answers: 0.159 +/- 0.15689216524464028
    Scores of predicted answers: 0.066 +/- 0.11121005695384194
    Positional bias occurrence on gold answers: 0.16
    Positional bias occurrence on predicted answers: 0.07

    Scores for criteria 'conciseness'
    Scores of gold answers: 1.0 +/- 0.0
    Scores of predicted answers: 0.34 +/- 0.47609522856952335
    Positional bias occurrence on gold answers: 0.03
    Positional bias occurrence on predicted answers: 0.01
