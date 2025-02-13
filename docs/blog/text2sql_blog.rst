.. _text-to-sql-evaluation-with-unitxt:

====================================================================================================
From Natural Language to Database Queries: Unleashing the Power of Text-to-SQL Evaluation with Unitxt
====================================================================================================

**Authors**: Yotam Perlitz, Elron Bandel  
**Date**: 2025-01-27

.. _introduction:

Introduction: Bridging the Gap Between Humans and Data – A New Era of Accessibility
------------------------------------------------------------------------------------

In today's data-saturated world, the ability to extract meaningful insights from databases is paramount. But for many, the complex language of SQL remains an imposing barrier. Imagine a world where you could simply ask a question in plain English and get the data you need—no SQL knowledge required. This is the promise of Text-to-SQL models: they act as translators, turning human language into database queries.

However, how can we be sure these models are accurately translating our intentions? How do we measure their effectiveness and ensure they generate reliable queries? This is where **Unitxt's latest addition** comes into play: comprehensive **Text-to-SQL evaluations** that address these challenges head-on. With this, developers can build more accurate, reliable, and user-friendly data access tools. In this post, we’ll unveil this feature and show you how to harness its full potential, unlocking a new level of data accessibility.

.. _challenges:

The Challenges of Text-to-SQL Evaluation: Why It's Harder Than You Think
--------------------------------------------------------------------------

Before diving into the specifics of Unitxt's new capabilities, it's crucial to understand the unique challenges of evaluating Text-to-SQL models. Unlike many other natural language processing tasks, evaluating Text-to-SQL models is more than just assessing whether the generated code is syntactically correct. The evaluation process is complex and multi-faceted, requiring a deep interaction with the underlying database.

Here are some key factors that make this evaluation process so challenging:

1. **Schema Fetching: Understanding the Database Structure**

   Models must understand the database structure by **fetching the schema** (tables, columns, relationships). This provides the context for generating valid and meaningful queries. Without accurate schema knowledge, the query might be syntactically correct but irrelevant or incorrect.

2. **Execution Accuracy: The Ultimate Test**

   The core of the evaluation is **execution accuracy**: does the query retrieve the *exact* data intended by the user's request? This requires executing the query against the database and comparing the results with the ground truth. This step demands a live database connection and precise output evaluation, adding significant complexity.

3. **Semantic Accuracy vs. Syntactic Correctness**

   A generated SQL query can be syntactically valid (it follows SQL rules and doesn't throw syntax errors) but still be completely wrong in terms of meaning. For example, consider the natural language question: "Which employees in the Sales department earn more than $50,000?". A model might generate a query that correctly selects employees and filters by salary but forgets to include the ``WHERE`` clause specifying the Sales department. The query would run, but it wouldn't answer the user's question.

4. **The "Needle in a Haystack" Problem**

   Even a tiny error in a SQL query can lead to drastically different results. A wrong join condition, a missing ``WHERE`` clause, or an incorrect aggregation can produce an output that's entirely off the mark. This makes rigorous evaluation crucial—we need to ensure the model is getting every detail right, not just most of them.

5. **Difficulties of Traditional Methods**

   Traditional Text-to-SQL evaluation often involved manually inspecting generated queries and their outputs. This process is incredibly time-consuming, prone to human error, and doesn't scale well as the number of test cases grows.

6. **The Live Database Dependency**

   Ultimately, you can't truly evaluate a Text-to-SQL model without executing its queries on a real database. Theoretical analysis or checking against a set of pre-defined rules is insufficient. You need to see if the query actually retrieves the correct data from the database it's intended for.

Text-to-SQL evaluation, therefore, isn’t just about code. It tests a model's ability to understand a database, interpret natural language, and reliably extract the desired information. This requires a sophisticated framework—one that Unitxt is proud to offer.

.. _introducing-unitxt-solution:

Introducing Unitxt's Text-to-SQL Evaluation: Automating the Complex
-------------------------------------------------------------------

Unitxt's new Text-to-SQL evaluation feature directly addresses the challenges outlined above. It provides an automated, standardized, and rigorous framework for evaluating models that translate natural language into SQL queries. Here's how Unitxt simplifies and enhances the evaluation process:

*   **Automated Execution:** Unitxt handles the tedious process of running generated SQL queries against the target database and comparing the results to the ground truth. No more manual inspection or writing custom scripts for each dataset.
*   **Schema-Awareness:** Unitxt understands database schemas. Through the card system (explained below), it fetches the schema information, ensuring that the generated queries are not just valid SQL but also relevant and meaningful within the context of the specific database structure.
*   **Execution Accuracy as the Core Metric:** Unitxt focuses on the most critical metric: **execution accuracy**. It doesn't just check if a query runs without errors; it verifies whether the query retrieves the *exact* data the user intended, based on a comparison with the ground truth results.
*   **Standardized and Reproducible:** Unitxt provides a consistent and objective evaluation framework. This is essential for fairly comparing different Text-to-SQL models, tracking progress during development, and ensuring that results are reproducible.

.. _unitxt-cards:

The Magic Behind the Scenes: How Unitxt's Cards Power Text-to-SQL Evaluation
-----------------------------------------------------------------------------

At the heart of Unitxt's Text-to-SQL evaluation capabilities lies the concept of **cards**. These cards are not just simple dataset loaders; they are comprehensive configurations that encapsulate all the necessary information for evaluating a Text-to-SQL model on a specific dataset and database.

Let's take a closer look at how the Text-to-SQL task is defined in Unitxt:

.. code-block:: python

    from typing import Optional

    from unitxt.blocks import Task
    from unitxt.catalog import add_to_catalog
    from unitxt.types import SQLDatabase

    add_to_catalog(
        Task(
            input_fields={
                "id": str,
                "utterance": str,
                "hint": Optional[str],
                "db": SQLDatabase,
            },
            reference_fields={"query": str},
            prediction_type=str,
            metrics=["metrics.text2sql.execution_accuracy", "metrics.anls"],
        ),
        "tasks.text2sql",
        overwrite=True,
    )

This code defines a ``Task`` called ``tasks.text2sql``. Notice how the ``input_fields`` include an ``SQLDatabase`` object. This is where the magic happens! The Text-to-SQL card provides the concrete details for this ``SQLDatabase`` object, specifying the database type, connection details, and schema. This allows Unitxt to automatically fetch the database schema and, crucially, execute the generated SQL queries during evaluation. The ``metrics`` field, including ``metrics.text2sql.execution_accuracy``, tells Unitxt how to assess the model's performance by running the queries and comparing the results to the ground truth.

Now, let's look at an example of a **real** Text-to-SQL card for the BIRD dataset:

.. code-block:: python

    from unitxt.card import TaskCard
    from unitxt.load import LoadHF
    from unitxt.operators import Shuffle
    from unitxt.task import Task
    import sys

    card = TaskCard(
        loader=LoadHF(path="premai-io/birdbench", split="validation"),
        preprocess_steps=[
            Shuffle(page_size=sys.maxsize),
        ],
        task="tasks.text2sql",
        templates="templates.text2sql.all",
    )

**Dissecting the Card:**

*   **``loader=LoadHF(path="premai-io/birdbench", split="validation")``:** This tells Unitxt to load the BIRD dataset from the Hugging Face Hub. The ``premai-io/birdbench`` is the dataset identifier, and ``validation`` specifies that we want to use the validation split of the dataset. The loader handles downloading the dataset and the associated database.
*   **``preprocess_steps=[Shuffle(page_size=sys.maxsize)]``:** This step shuffles the dataset to ensure randomness during evaluation.
*   **``task="tasks.text2sql"``:** This connects the card to the general Text-to-SQL task definition we saw earlier. It tells Unitxt that this card is for evaluating Text-to-SQL models. The task definition includes the crucial `SQLDatabase` type in its input fields, indicating the need for a database and schema information.
*   **``templates="templates.text2sql.all"``:** This specifies that we want to use all available templates for formatting the natural language input into prompts for the Text-to-SQL model.

**How the Card Enables Evaluation:**

This card, in conjunction with the ``tasks.text2sql`` definition, provides Unitxt with all the necessary information to perform the evaluation:

1. **Dataset and Database:** The ``loader`` fetches the BIRD dataset, which includes the SQLite database file.
2. **Schema:** Unitxt can automatically extract the database schema from the SQLite file. This schema information (tables, columns, relationships) is crucial for understanding the context of the generated queries.
3. **Task Definition:** The ``task`` field links to the ``tasks.text2sql`` definition, which specifies that ``execution_accuracy`` is the primary metric.
4. **Execution:** During evaluation, Unitxt uses the schema information and the connection details (implicit in the SQLite file) to execute the generated SQL queries against the BIRD database.
5. **Comparison:** The results of the executed queries are then compared to the ground truth results to calculate the ``execution_accuracy``.

In essence, this card encapsulates all the complexities of setting up the evaluation environment, allowing you to focus on developing and testing your Text-to-SQL models.

.. _hands-on-example:

Putting it into Practice: Evaluating Llama-3-70b on BIRD with Unitxt
---------------------------------------------------------------------

Let's see how to use Unitxt to evaluate a Text-to-SQL model on the BIRD benchmark. We'll use the powerful ``llama-3-3-70b-instruct`` model for this demonstration, but remember, Unitxt allows you to easily swap in different models.

.. code-block:: python

    from unitxt import evaluate, load_dataset, settings
    from unitxt.inference import CrossProviderInferenceEngine
    from unitxt.text_utils import print_dict

    # 1. Using the 'cards.text2sql.bird' card: This card tells Unitxt everything it needs to know about the BIRD dataset, including:
    #    - Where to download the BIRD database.
    #    - How to connect to the database and fetch its schema.
    #    - How to evaluate the generated SQL queries (using execution accuracy).

    with settings.context(
        disable_hf_datasets_cache=False,
        allow_unverified_code=True,
    ):
        test_dataset = load_dataset(
            "card=cards.text2sql.bird",
            "template=templates.text2sql.you_are_given_with_hint_with_sql_prefix",
            split="validation",
        )

    # 2. Setting up the model (standard Unitxt inference)
    inference_model = CrossProviderInferenceEngine(
        model="llama-3-3-70b-instruct",
        max_tokens=256,
    )

    # 3. Generating predictions (standard Unitxt inference)
    predictions = inference_model.infer(test_dataset)

    # 4. Evaluation: This is where the magic happens!
    #    - The 'evaluate' function uses the information from the 'cards.text2sql.bird' card.
    #    - It executes the predicted SQL queries on the BIRD database.
    #    - It compares the results to the ground truth to calculate the 'execution_accuracy'.
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    # 5. Examining the results
    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "prediction",
            "subset",
        ],
    )
    print_dict(
        evaluated_dataset[0]["score"]["global"]["execution_accuracy"],
    )

**Explanation:**

1. **Loading the Dataset with the Card:** We use ``load_dataset`` with ``"card=cards.text2sql.bird"``. This is the crucial step. The ``cards.text2sql.bird`` card, which we defined above, contains all the instructions for:
    *   Downloading the BIRD database (if it's not already present).
    *   Connecting to the database and automatically fetching its schema (tables, columns, relationships).
    *   Knowing that ``execution_accuracy`` is the metric to use for evaluation.

    The ``template`` parameter specifies how to format the input for the model.

2. **Setting Up the Model:** We initialize the ``CrossProviderInferenceEngine`` with the ``llama-3-3-70b-instruct`` model. This part is standard Unitxt inference setup.

3. **Generating Predictions:** The ``inference_model.infer(test_dataset)`` line runs the model on the dataset, generating SQL query predictions.

4. **Evaluation:** The ``evaluate`` function is where the core Text-to-SQL evaluation happens. Guided by the ``cards.text2sql.bird`` card, it:
    *   Takes the generated SQL queries.
    *   Executes them on the BIRD database (using the connection details and schema information from the card).
    *   Compares the execution results with the ground truth to calculate the ``execution_accuracy``.

5. **Interpreting the Results:** The code then prints the results, including the ``execution_accuracy`` score. This score tells us how well the model performed in generating SQL queries that retrieve the *exact* expected data from the BIRD database. The assert statement checks if the score is above a certain threshold, indicating good performance.

.. _database-support:

Diverse Database Support: Local, Remote, and In-Memory
------------------------------------------------------

Unitxt's flexible framework supports three different types of database environments:

1. **Local Databases**: Automatically handles downloading and setup of databases for datasets like BIRD, as shown in the previous example.
2. **Remote Databases**: Enables connection to external data sources via API, allowing evaluation on live, dynamic data. 
3. **In-Memory Databases**: Allows defining databases directly in code using dictionaries—ideal for custom datasets or sensitive data. 

This versatility ensures that you can evaluate your Text-to-SQL models in a way that best suits your needs. Whether you're testing on a local database, pulling data from a live source, or working with in-memory setups, Unitxt adapts to your scenario.

.. _beyond-bird:

Beyond BIRD: A Universal Solution for Text-to-SQL Evaluation Across Diverse Datasets
-----------------------------------------------------------------------------------

While the example above showcases the power of Unitxt on the BIRD dataset, it's important to note that this is just the beginning. Unitxt's evaluation framework is designed to be a **universal solution**, supporting a wide array of Text-to-SQL datasets, including:

*   **Spider**: A large-scale, complex, cross-domain dataset widely used as a benchmark for semantic parsing and Text-to-SQL tasks.
*   **FIBEN**: A challenging new benchmark focused on financial data, pushing the boundaries of Text-to-SQL models in this domain.
*   **And many more!** We're continuously expanding our support for new datasets.

Thanks to Unitxt’s modular design, switching between datasets is a breeze. Simply adjust the ``card`` parameter in the ``load_dataset`` function to seamlessly work with different datasets tailored to your specific evaluation needs and research goals.

.. _conclusion:

Conclusion: Shaping the Future of Data Interaction – Empowering Everyone with Data
----------------------------------------------------------------------------------

Unitxt’s groundbreaking Text-to-SQL evaluation feature is a game-changer for developers working on models that aim to translate natural language into SQL queries. By providing an automated, standardized, and rigorous evaluation framework, Unitxt dramatically accelerates the development of more accurate, reliable, and user-friendly Text-to-SQL systems.

We invite you to dive into this exciting new feature and join us on this journey to shape the future of data interaction. With Unitxt, you can unlock the true potential of your data, making it more accessible than ever before. Empower everyone—regardless of their SQL expertise—to effortlessly query, explore, and understand the wealth of information hidden within databases.

**References**: `BIRD Benchmark <https://bird-bench.github.io/>`__