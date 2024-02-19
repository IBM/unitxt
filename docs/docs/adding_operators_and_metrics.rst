.. _adding_operator:

=====================================
Adding Stream Operators and Metrics
=====================================

In this section we will add brand new stream operators and metrics
to use in our processing pipelines.

Adding a new stream operator
----------------------------

Create a new class that extends the `StreamInstanceOperator` or any other :ref:`Stream Operator <operators>` class.

    .. code-block:: python

        from unitxt.operator import StreamInstanceOperator

        class AddFields(StreamInstanceOperator):
            fields: Dict[str, object]

            def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
                return {**instance, **self.fields}

To test that our operator works as expected, we can use the unitxt built-in
testing suit:

    .. code-block:: python

        from unitxt.test_utils.operators import check_operator

        operator = AddFields(fields={"b": 2})

        inputs = [{'a': 1}, {'a': 2}]
        targets = [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}]

        print(check_operator(operator, inputs, targets)) # True


Adding a new metric
-------------------

Create a new class that extends the `Metric` or any other :ref:`Metric <metrics>` class.

    .. code-block:: python

        class Accuracy(SingleReferenceInstanceMetric):
            reduction_map = {"mean": ["accuracy"]}
            main_score = "accuracy"

            def compute(self, reference, prediction: str) -> dict:
                return {"accuracy": float(str(reference) == str(prediction))}

Other base classes for metrics are: `InstanceMetric`, `GlobalMetric`.

To test our metric work as expected we can use unitxt built in
testing suit:

    .. code-block:: python

        from unitxt.test_utils.metrics import test_metric

        metric = Accuracy()

        predictions = ['positive', 'negative']
        references = [['positive'], ['positive']]
        target = {'accuracy': 0.5}

        print(test_metric(metric, predictions, references, target)) # True
