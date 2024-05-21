.. _data_classification_policy

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Sensitive data in unitxt ✨
=====================================

The section discusses how to properly handle sensitive data in unitxt in order to avoid accidentally sending proprietary/confidential data to external services.

Data classification policy
----------------------------
Whenever processing streams in unitxt, users can specify a parameter called `data_classification_policy`, which determines how this data should be treated. The taxonomy is defined by a user and may encompass multiple different policies.

Each component used in Unitxt (metrics, operators, inference engines etc.) has the same parameter as well. This allows a given component to verify if it can process instances of a stream.

If user-defined policies for a component include that of data, then a stream may be further processed. Otherwise, an error will be raised.

The purpose is to ensure that potentially sensitive data (for example PII information) is handled in a proper way. That is particularly important when accessing external services, for instance calling metrics which are calculated remotely.

The parameter itself should be a list of strings, which are names of considered policies.

Adding `data_classification_policy` for data
----------------------------

Data classification information is added to streams of data by the use of unitxt loaders.

Users need to set the `data_classification_policy` parameter of a chosen loader, which value will be then added as an additional field to all instances within a stream.

Example:

.. code-block:: python

    from unitxt.loaders import LoadFromDictionary

    data = {
        "train": {"text": "SomeText1", "output": "SomeResult1"},
        "test": {"text": "SomeText2", "output": "SomeResult2"},
    }

    loader = LoadFromDictionary(
        data=data,
        data_classification_policy=["public"],
    )

    multi_stream = loader.process()  # the field will be added during processing

    assert multi_stream["test"]["data_classification_policy"] == ["public"]

Adding `data_classification_policy` for components
----------------------------

In case of unitxt components, the parameter can be added by either setting an environment variable or setting the attribute of a class in the code.

1. **Using env**:

To specify data classification policy for a chosen unitxt component, you need to set the `UNITXT_DATA_CLASSIFICATION_POLICY` env variable accordingly. It should be of type `Dict[str, List[str]]`, where a key is a name of a given artifact, and a corresponding value is a list of applied policies. For example:

.. code-block:: python

    UNITXT_DATA_CLASSIFICATION_POLICY = {
        "metrics.accuracy": ["public"],
        "templates.span_labeling.extraction.carry": ["pii", "propriety"],
    }

2. **Setting class attribute**:

The other way is to set the `data_classification_policy` attribute of a given artifact directly inside the code. Again, it should be a list of string, and its default value is None.

It is important to keep in mind that the priority always go to a value present in env and a component will try to use it first. If the env variable was not configured, then the passed value of `data_classification_policy` is used instead.

Example:

.. code-block:: python

    from unitxt.metrics import F1Binary
    from unitxt.operators import DuplicateInstances

    stream = [
        {"input": "Input1", "data_classification_policy": ["pii", "propriety"]},
        {"input": "Input2", "data_classification_policy": ["pii", "propriety"]},
    ]

    metric = F1Binary(data_classification_policy=["public"])
    metric.process(stream)  # will raise an error as policies are different

    operator = DuplicateInstances(
        num_duplications=2,
        data_classification_policy=["pii"],
    )
    operator.process_instance(stream[0])  # will not raise an error as the policy is included
