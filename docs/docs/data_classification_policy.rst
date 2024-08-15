.. _data_classification_policy

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Sensitive data in unitxt ✨
=====================================

The section discusses how to properly handle sensitive data in Unitxt in order to avoid accidentally exposing 
proprietary/confidential/personal data to unauthorized services or 3rd parties. For example, sending sensitive 
data for inference by an external API in LLM as Judge metric.

The problem is exacerbated since the person who owns the data and uses the metric in their card
may not know what 3rd party services are used internally by the metric.

To address this, Unitxt allows the data owner to specify the data classification of their data, and similarly it requires that
any metric (or other component) that processes the data must be explicitly allowed to process data with this classification.


Data classification policy
----------------------------

When data is loaded from an external data source using a Loader, it can be tagged with a `data_classification_policy`,
which is a list of string identifiers, such as `public`, `proprietary`, `confidential`, `pii`.
You can define your own data classification identifiers.

Each component that processes data in Unitxt ( operators, metrics, inference engines, etc.) also has 
a parameter called `data_classification_policy`.  This parameter determines which kinds of data
it can process.  The parameter is also a list of string identifiers, each of which is a name of allowed data classification.

Before processing the data, the component verifies that the `data_classification_policy` of the data meets its `data_classification_policy`.
If the policies for a component include the classification of the data, then the data may be further processed. Otherwise, an error will be raised.
For example, an LLM as judge that calls an external api may set `data_classification_policy` to `['public']`.
If data marked [`confidential`] is passed to the metric, it will not process the data and fail.

If the data has multiple values under `data_classification_policy` then the component must be allowed to handle all of them.
If the `data_classification_policy` is not set, the component can handle all data.  

It is possible to override the `data_classification_policy` of a component with an environment variable.  See below.

Adding `data_classification_policy` for data
----------------------------

Data classification information is added to streams of data by the use of Unitxt loaders.
Existing loaders have default data classification policies. For example, LoadHF sets the policy to `['public']` for datasets
downloaded from the HuggingFace and `['proprietary']` for datasets loaded from local files.  You can override this by setting
the `data_classification_policy` parameter of the loader. 

The data classification value is added as an additional field to all instances within a stream.

Example:

.. code-block:: python

    from unitxt.loaders import LoadFromDictionary

    data = {
        "train": [{"text": "SomeText1", "output": "SomeResult1"}],
        "test": [{"text": "SomeText2", "output": "SomeResult2"}],
    }

    loader = LoadFromDictionary(
        data=data,
        data_classification_policy=["public"], # Overrides the default of ["proprietary"]
    )

    multi_stream = loader.process()  # the field will be added during processing
    dataset = multi_stream.to_dataset()
    assert dataset["test"][0]["data_classification_policy"] == ["public"]

Adding `data_classification_policy` for components
----------------------------

In case of Unitxt components, the parameter can be added by setting the attribute of a class in the code or by setting an environment variable.

1. **Setting default data classification policy class attribute**:

The `data_classification_policy` attribute can be set in the code when the class is created.
The attribute should be a list of strings, and its default value is None.

Example:

.. code-block:: python

    from unitxt.metrics import F1Binary
    from unitxt.operators import DuplicateInstances

    stream = [
        {"input": "Input1", "data_classification_policy": ["pii", "proprietary"]},
        {"input": "Input2", "data_classification_policy": ["pii", "proprietary"]},
    ]

    metric = F1Binary(data_classification_policy=["public"])
    list(metric.process(stream))  # will raise an error as policies are different

    operator = DuplicateInstances(
        num_duplications=2,
        data_classification_policy=["pii"],
    )
    list(operator.process(stream))  # will not raise an error as the policy is included


1. **Overriding default policy during environment variable **:


You can override the data classification of artifacts that was saved in the catalog by setting the `UNITXT_DATA_CLASSIFICATION_POLICY` env variable accordingly.
It should be a string representation of type `Dict[str, List[str]]`, where a key is a name of a given artifact, and a corresponding value is the allowed data classification. For example:

.. code-block:: bash

    export UNITXT_DATA_CLASSIFICATION_POLICY '{ "metrics.llm_as_judge.rating.mistral_7b_instruct_v0_2_huggingface_template_mt_bench_single_turn": ["public","proprietary", "pii"], "processors.translate": ["public", "proprietry"]}'



