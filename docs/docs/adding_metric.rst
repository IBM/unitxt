.. _adding_metric:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.


=====================================
Adding Metrics ✨
=====================================

In this section we will add brand new stream operators and metrics
to use in our processing pipelines.


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
