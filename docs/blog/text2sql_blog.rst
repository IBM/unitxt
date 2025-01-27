From Natural Language to Database Queries: Unleashing the Power of Text-to-SQL Evaluation with Unitxt
====================================================================================================

**Authors**: Yotam Perlitz, Elron Bandel  
**Date**: 2025-01-27

Introduction: Bridging the Gap Between Humans and Data – A New Era of Accessibility
------------------------------------------------------------------------------------

In today's data-saturated world, the ability to extract meaningful insights from databases is paramount. But for many, the complex language of SQL remains an imposing barrier. Imagine a world where you could simply ask a question in plain English and get the data you need—no SQL knowledge required. This is the promise of Text-to-SQL models: they act as translators, turning human language into database queries.

However, how can we be sure these models are accurately translating our intentions? How do we measure their effectiveness and ensure they generate reliable queries? This is where **Unitxt's latest addition** comes into play: comprehensive **Text-to-SQL evaluations** that addresses these challenges head-on. With this, developers can build more accurate, reliable, and user-friendly data access tools. In this post, we’ll unveil this feature and show you how to harness its full potential, unlocking a new level of data accessibility.

The Hurdles of Text-to-SQL Evaluation: Beyond Code Generation
-------------------------------------------------------------

Before diving into the specifics of the new Unitxt’s capabilities, it's important to understand the unique challenges of evaluating Text-to-SQL models. Unlike many other natural language processing tasks, evaluating Text-to-SQL models is more than just assessing whether the generated code is syntactically correct. The evaluation process is complex and multi-faceted, requiring a deep interaction with the underlying database.

Here are two key factors that make this evaluation process so challenging:

1. **Schema Fetching: Understanding the Database Structure**

   Models must understand the database structure by **fetching the schema** (tables, columns, relationships). This provides the context for generating valid and meaningful queries. Without accurate schema knowledge, the query might be syntactically correct but irrelevant or incorrect.

2. **Execution Accuracy: The Ultimate Test**

   The core of the evaluation is **execution accuracy**: does the query retrieve the *exact* data intended by the user's request? This requires executing the query against the database and comparing the results with the ground truth. This step demands a live database connection and precise output evaluation, adding significant complexity.

Text-to-SQL evaluation, therefore, isn’t just about code. It tests a model's ability to understand a database, interpret natural language, and reliably extract the desired information. This requires a sophisticated framework—one that Unitxt is proud to offer.

The Challenge: Evaluating Text-to-SQL Models – Beyond Simple Accuracy
--------------------------------------------------------------------

Evaluating Text-to-SQL models presents unique challenges that go beyond traditional tasks. Unlike tasks with straightforward answers, a "correct" SQL query is multifaceted. It must be syntactically correct (following SQL rules) and semantically accurate (capturing the user's intent precisely). It must execute flawlessly without errors, and most importantly, it must retrieve *exactly* the data the user is looking for. Even a minor error in the query can lead to vastly different results, making rigorous and thorough evaluation absolutely critical.

Traditional evaluation methods often involve tedious manual inspection of generated queries and their outputs—a process that's both time-consuming and prone to inconsistencies. Unitxt’s new evaluation capabilities automate this entire process, leveraging standardized metrics to meticulously assess the quality of generated SQL queries. This automation saves valuable time and resources, ensuring consistent and objective evaluation across the board.

Text-to-SQL Evaluation with Unitxt: A Hands-On Example with the BIRD Benchmark
-------------------------------------------------------------------------------

Let’s dive into a practical, hands-on example to see Unitxt's text-to-SQL evaluation in action. Below is a code snippet that showcases the evaluation of a text-to-SQL model on the BIRD benchmark—a dataset specifically designed to test models on complex database schemas and challenging natural language queries. We’ll be using the powerful ``llama-3-3-70b-instruct`` model for this demonstration, but remember, Unitxt offers the flexibility to easily swap in different models based on your needs.

.. code-block:: python

    from unitxt import evaluate, load_dataset, settings
    from unitxt.inference import CrossProviderInferenceEngine
    from unitxt.text_utils import print_dict

    with settings.context(
        disable_hf_datasets_cache=False,
        allow_unverified_code=True,
    ):
        test_dataset = load_dataset(
            "card=cards.text2sql.bird",
            "template=templates.text2sql.you_are_given_with_hint_with_sql_prefix",  # Consider setting loader_limit for faster testing
            split="validation",
        )

    # Infer
    inference_model = CrossProviderInferenceEngine(
        model="llama-3-3-70b-instruct",
        max_tokens=256,
    )

    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "prediction",
            "subset",
        ],
    )
    print_dict(
        evaluated_dataset[0]["score"]["global"],
    )

    assert (
        evaluated_dataset[0]["score"]["global"]["execution_accuracy"] >= 0.44
    ), f'Results degraded: metric is below threshold, received score {evaluated_dataset[0]["score"]["global"]["score"]}'

Let’s break down this code step-by-step to understand the magic happening behind the scenes:

1. **Loading the Dataset: Setting the Stage**

   We kick things off by loading the [BIRD validation dataset]_ using the ``load_dataset`` function. Here, we specify the dataset card (``cards.text2sql.bird``) and the template (``templates.text2sql.you_are_given_with_hint_with_sql_prefix``). This template acts as a guide, instructing Unitxt on how to format natural language queries into well-structured SQL prompts for the model. **Note** that this is where much of the magic happens—when the template is rendered, the database is accessed, and the schema is fetched.

2. **Setting Up Inference: Choosing Your Powerhouse**

   The ``CrossProviderInferenceEngine`` is our command center for handling inference with the chosen model, in this case, ``llama-3-3-70b-instruct``. This engine provides incredible flexibility, allowing you to seamlessly switch between different models and providers.

3. **Generating Predictions: From Natural Language to SQL**

   The command ``inference_model.infer(test_dataset)`` runs inference on the test dataset, prompting the model to generate SQL query predictions based on natural language queries.

4. **Evaluating Performance: The Moment of Truth**

   Unitxt's powerful ``evaluate`` function steps in, comparing the generated SQL queries against the ground truth. It calculates critical metrics like **execution accuracy**, which measures whether the query not only executes without errors but also returns the precise results expected. **Execution accuracy** is far from a trivial metric; it includes the live database connection and careful result comparison.

Diverse Database Support: Local, Remote, and In-Memory
------------------------------------------------------

Unitxt’s flexible framework supports three different types of database environments:

1. **Local Databases**: Automatically handles downloading and setup of databases for datasets like BIRD, as shown in the previous example.
2. **Remote Databases**: Enables connection to external data sources via API, allowing evaluation on live, dynamic data. To use a remote database, inherit from the ``RemoteDB`` class and implement the ``_execute`` method.
3. **In-Memory Databases**: Allows defining databases directly in code using dictionaries—ideal for custom datasets or sensitive data. Simply assign the DB to the ``in_memory_db`` field in the ``LoadDB`` class.

This versatility ensures that you can evaluate your Text-to-SQL models in a way that best suits your needs. Whether you're testing on a local database, pulling data from a live source, or working with in-memory setups, Unitxt adapts to your scenario.

Beyond BIRD: A Universal Solution for Text-to-SQL Evaluation Across Diverse Datasets
-----------------------------------------------------------------------------------

While the example above showcases the power of Unitxt on the BIRD dataset, it's important to note that this is just the beginning. Unitxt's evaluation framework is designed to be a **universal solution**, supporting a wide array of Text-to-SQL datasets, including:

- **Spider**: A large-scale, complex, cross-domain dataset widely used as a benchmark for semantic parsing and Text-to-SQL tasks.
- **FIBEN**: A challenging new benchmark focused on financial data, pushing the boundaries of Text-to-SQL models in this domain.
- **And many more!** We're continuously expanding our support for new datasets.

Thanks to Unitxt’s modular design, switching between datasets is a breeze. Simply adjust the ``card`` parameter in the ``load_dataset`` function to seamlessly work with different datasets tailored to your specific evaluation needs and research goals.

Conclusion: Shaping the Future of Data Interaction – Empowering Everyone with Data
----------------------------------------------------------------------------------

Unitxt’s groundbreaking Text-to-SQL evaluation feature is a game-changer for developers working on models that aim to translate natural language into SQL queries. By providing an automated, standardized, and rigorous evaluation framework, Unitxt dramatically accelerates the development of more accurate, reliable, and user-friendly Text-to-SQL systems.

We invite you to dive into this exciting new feature and join us on this journey to shape the future of data interaction. With Unitxt, you can unlock the true potential of your data, making it more accessible than ever before. Empower everyone—regardless of their SQL expertise—to effortlessly query, explore, and understand the wealth of information hidden


[BIRD validation dataset]_ https://bird-bench.github.io/