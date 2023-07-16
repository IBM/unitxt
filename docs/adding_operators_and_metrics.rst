
=====================================
Adding Stream Operators and Metrics
=====================================

In this section we will add brand new stream operators and metrics
to use in our processing pipelines. 

Adding a new stream operator
----------------------------

Create a new class that extends the `StreamInstanceOperator` or any other :ref:`Stream Operator <operators>` class.

    .. code-block:: python

        class AddFields(StreamInstanceOperator):
            fields: Dict[str, object]

            def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
                return {**instance, **self.fields}


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