.. _adding_metric:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.


=====================================
Metrics ✨
=====================================

Unitxt supports a large collection of built in metrics, from classifical ones such as
rouge, bleu, f1 to embedding based score like SentenceBert and Bert score, as well as
llm as judges using local or API based models.

You specify the metrics metrics in the Task.

For example:

.. code-block:: python

    task = Task(
            input_fields={"question" : str},
            reference_fields={"answer" : str},
            prediction_type=str,
            metrics=[
                "metrics.rouge",
                "metrics.normalized_sacrebleu",
                "metrics.bert_score.deberta_xlarge_mnli",
                "metrics.bert_score.deberta_large_mnli"
            ],
    )

The full list of built in metrics shows in the :ref:`Metrics part of the catalog <dir_catalog.metrics>`.
In this section we will understand Unitxt metrics and learn how to add new metrics.


Metric Inputs
-------------

Unitxt metrics receive three inputs for each instance:

1. **Prediction** (``prediction``):  The prediction passed to the metric is not the raw textual prediction
returned by the model, but rather the processed prediction, after applying the post processors
defined by the template.  The role of the template's post processors is to convert the output
of the model to the required type of the metrics.  For example, a spearman metric expects a float
prediction.  A post processor in the template will cast the string input to a float, and return NaN
if the string can not be converted to a float.  Another example, a multi-label f1 expects a list of
string class names as predictions.   The post processor may convert the string output into a list
(e.g. by splitting using a separator).

2. **References** (``references`` - optional):  This is a list of gold references, from the same type of the prediction.
For example, if the prediction is a string, the references field is a list of strings.  If the prediction is
a list of strings (e.g in multi-label classification), then the references field is a *list* of lists of strings.
The metric should return a perfect score, if the prediction is equal to one of the references.

3. **Task data** (``task_data`` - optional) - all the input and output fields of a task as a dictionary.
The input fields can be used to create reference-less metrics.



In the rest of the section, we will assume we want to create a new metric for the
task of calculating of the sum of integers (see  :ref:`adding task <task>`.)

It's important that all processing will be done in the template's post processor and not in the metric,
because different templates may require different processing.  For example, one template may request
the model response as a number (e.g. "35") or request a model response in words (e.g. "thirty five").
The metric should receive a single integer.

Metric Outputs
--------------

By default, each metric provides scores for each instance separately and global aggregated scores over all instances together.
The output of the metrics is a nested dictionary per instance.

The scores calculated on instance ``i`` by itself are found in ``results[i]["score"]["instance"]``.
The global scores calculated over all instances are found in ``results[i]["score"]["global"]``.
Note the global scores are the same in all instances, so typically, ``results[0]["score"]["global"]`` is used to get the global scores.

A metric could return multiple scores, but it should always return a field called ``score`` with the main score of the metric,
and ``score_name`` which is the name of the main score.

For example, the score list for an instance could be:

.. code-block:: python

    {
        "sum_accuracy_approximate": 0.0,
        "score": 1.0,
        "score_name": "sum_accuracy_approximate"
    }

The global scores are calculated over all instances.

Metrics can also calculate confidence intervals for the global scores.
This gives you an assessment of the inherient noise in the scores.  When you compare runs on same data, check if their confidence
intervals overlap. If so, the difference may not be statistically significant.

.. code-block:: python

    {
        "sum_accuracy_approximate": 0.67,
        "score": 0.67,
        "score_name": "sum_accuracy_approximate",
        "sum_accuracy_approximate_ci_low": 0.53,
        "sum_accuracy_approximate_ci_high": 0.83,
        "score_ci_low": 0.53,
        "score_ci_high": 0.83,
    }

Metric Outputs with Multiple Metrics
------------------------------------

When multiple metrics are specified, their scores are appended to the score list.
If multiple metrics have the same score names, the score of the metric that appears first in the metrics list has precedence.

If you want to avoid the scores being overwritten by other metrics, you can add a prefix to each metric score.

.. code-block:: python

    task = Task(
        ...
        metrics=[
            "metrics.rouge",
            "metrics.normalized_sacrebleu",
            "metrics.bert_score.deberta_xlarge_mnli[score_prefix=sbert_deberta_xlarge_mnli_]",
            "metrics.bert_score.deberta_large_mnli[score_prefix=sbert_deberta_large_mnli_]"
            ],
    )

Note that the ``score`` and ``score_name`` are always taken from the first metric in the ``metrics`` list.

Metric Base Classes
-------------------

As described in the previous section, a metric generates a set of scores per instance (called ``instance`` scores),
and a set of scores over all instances (called ``global`` scores).

Unitxt has several base classes, subclasses of class :class:`Metric <unitxt.metric.Metric>`, that simplify the creation 
of metrics, depending on how the scores are calculated.

:class:`InstanceMetric <unitxt.metrics.InstanceMetric>` - Class for metrics in which the global scores are calculated by aggregating the instance scores.
Typically, the global score is the average of all instance scores. :class:`InstanceMetric <unitxt.metrics.InstanceMetric>` first evaluates each instance separately,
and then aggregates the scores of the instances. Some examples of instance metrics are ``Accuracy``, ``TokenOverlap``, ``CharEditDistance``.

:class:`BulkInstanceMetric <unitxt.metrics.BulkInstanceMetric>` - Similar to :class:`InstanceMetric <unitxt.metrics.InstanceMetric>`, it is for metrics 
in which the global score can be calculated by aggregating over the instance scores.  However,
for the sake of efficient implementation, it's better to run them in bulks (for example, when using LLMs during score calculations).
A ``BulkInstanceMetric`` calculates the instance scores of a batch of instances each time, but then aggregates over the scores of all the instances.
Some examples of bulk instance metrics are ``SentenceBert``, ``Reward``.

:class:`GlobalMetric <unitxt.metrics.GlobalMetric>` - Class for metrics for which the global scores must be calculated over all the instances together.
Some examples of global metrics are ``f1``, ``Spearman``, ``Kendall Tau``.  Note that by default, global metrics are executed once per instance
to generate per instance scores, and then once again over all instances together. So if there are 100 instances,
it will first be called 100 times, each on a single instance, and then one time on all 100 instances.

Instance scores of ``GlobalMetrics`` are useful for error-analysis. Consider ``f1`` score, for example.
It can be calculated only on all instances together. Yet it is useful to report the score of every instance
so you can see that good instances get ``f1`` score of 1 and bad ones get 0.

   .. note::
    By default global metrics are also executed once per instance as list (of size one),
    to generate per instance scores that are useful for debugging and sanity checks.

Adding a New Instance metric
----------------------------

Assume we want to create a referenceless metric for the task of adding two numbers.
It will take the processed prediction of the task (an integer) and compare to the sum of the
two task input fields ``num1`` and ``num2``.  It will check, for each instance,
how close the predicted sum is to the actual sum.
The metric can be configured with a ``relative_tolerance`` threshold for approximate comparison.
If the difference between the prediction and actual result is smaller than the ``relative_tolerance``
threshold, the instance score is 1. Otherwise, the instance result is 0.
The global accuracy result is the mean of the instance scores.

.. code-block:: python

    class SumAccuracy(InstanceMetric):

        main_score = "sum_accuracy" # name of the main score
        reduction_map = {"mean": ["sum_accuracy"]} # defines that the global score is a mean of the instance scores
        ci_scores = ["sum_accuracy"] # define that confidence internal should be calculated on the score
        prediction_type = int      # the metric expect the prediction as an int

        # Relation tolerance for errors by default it is 0, but can be changed for approximate comparison
        relative_tolerance : float = 0

        def compute(
            self, references: List[int], prediction: int, task_data: List[Dict]
        ) -> dict:
            actual_sum = task_data["num1"] + task_data["num2"]
            isclose_enough =  isclose(actual_sum, prediction, rel_tol=self.relative_tolerance)
            result = { self.main_score : 1.0 if isclose_enough else 0.0}
            return result

To verify that our metric works as expected we can use unitxt built in testing suit:

.. code-block:: python

    #
    # Test SumAccuracy metric and add to catalog
    #

    from unitxt_extension_example.metrics import SumAccuracy
    metric = SumAccuracy()

    predictions = [3, 799 , 50]
    references = [[5],[800],[50]]
    task_data = [{"num1" : 2, "num2" : 3}, {"num1" : 300, "num2" : 500}, {"num1" : -25, "num2" : 75}]
    instance_targets = [
        {"sum_accuracy": 0.0, "score": 0.0, "score_name": "sum_accuracy"},
        {"sum_accuracy": 0.0, "score": 0.0, "score_name": "sum_accuracy"},
        {"sum_accuracy": 1.0, "score": 1.0, "score_name": "sum_accuracy"},
    ]

    global_target = {
        "sum_accuracy": 0.33,
        "score": 0.33,
        "score_name": "sum_accuracy",
        "sum_accuracy_ci_low": 0.0,
        "sum_accuracy_ci_high": 1.0,
        "score_ci_low": 0.0,
        "score_ci_high": 1.0,
    }

    outputs = test_metric(
        metric=metric,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
        task_data=task_data
    )

    add_to_catalog(metric, "metrics.sum_accuracy")

Adding a Global Metric
----------------------

Now let's consider a global reference based metric that checks if accuracy depends on the magnitude of the results.
For example, is more accurate when the result is 1 digits vs 5 digits.
To check this, we will see if there is a correlation between the number of digits in the reference value and the accuracy.
This is a global metric because it performs the calculation over all the instance predictions and references together.

.. code-block:: python

    class SensitivityToNumericMagnitude(GlobalMetric):
    """
    SensitiveToNumericMagnitude is a reference-based metric that calculates if accuracy depends
    on the numeric magnitude of the reference value.  It receives integer prediction values and integer reference values
    and calculates the correlation between the number of digits in the reference values and the accuracy
    (whether predictions=references).

    The score is negative (up to -1), if predictions tend to be less accurate when reference values are larger.
    The score is close to 0, if the magnitude of the reference answer does not correlate with accuracy.
    The score is positive (up to 1), if predictions tend to be less accurate when reference values are smaller.

    In most realistic cases, the score is likely to be zer or negative.

    """
    prediction_type = int
    main_score="sensitivity_to_numeric_magnitude"
    single_reference_per_prediction = True  # validates only one reference is passed per prediction

    def compute(
        self, references: List[List[int]], predictions: List[int], task_data: List[Dict]
    ) -> dict:
        import scipy.stats as stats # Note the local import to ensure import is required only if metric is actually used
        magnitude = [ len(str(abs(reference[0]))) for reference in references ]
        accuracy = [ reference[0] == prediction  for (reference, prediction) in zip(references, predictions) ]
        spearman_coeff, p_value =  stats.spearmanr(magnitude, accuracy)
        result = { self.main_score :  spearman_coeff }
        return result



1. Calculating confidence intervals for global metrics can be costly if each invocation of the metric takes a long time.
To avoid calculating confidence internals for global metrics set ``n_resamples = 0``.

2. Unitxt calculates instance results in global metrics to allow viewing the output on a single instances.
This can help ensure metric behavior is correct, because it can be checked on single instance.
However, sometimes it does not make sense because the global metric assumes a minimum amount of instances.
The per instance calculations can be disabled by setting ``process_single_instances = False``.

Managing Metric Dependencies
----------------------------

If a metric depends on an external package (beyond the unitxt dependencies),
use of ``_requirements_list`` allows validating the package is installed and provides instructions to the users if it is not.

.. code-block:: python

    _requirements_list = { "sentence_transformers" : "Please install sentence_transformers using  'pip install -U sentence-transformers'" }

To ensure the package is imported only if the metric is actually used, include the import inside the relevant methods and not in global scope of the file.

Using Metric Pipelines
----------------------

Unitxt metrics must be compatible with the task they are used with.  However, sometime there is an implementation
of a metric that performs the needed calculations but expects different inputs.
The :class:`MetricPipeline <unitxt.metrics.MetricPipeline>` is a way to adapt an existing metric to a new task.
For example, the :class:`TokenOverlap <unitxt.metrics.TokenOverlap>` metric takes a string input prediction and a string references and calculates
the token overlap between them. If we want to reuse it, in a ``Retrieval Augmented Generation`` task to measure the token
overlap between the predictions and the context, we can define a ``MetricPipeline`` to copy the ``context`` field of the task
to the ``references`` field.  Then it runs the existing metric. Finally, it renames the scores to more meaningful names.

.. code-block:: python

    metric = MetricPipeline(
        main_score="score",
        preprocess_steps=[
            Copy(field="task_data/context", to_field="references"),
            ListFieldValues(fields=["references"], to_field="references"),
        ],
        metric="metrics.token_overlap",
        postprocess_steps=[
            Rename(
                field_to_field=[
                    ("score/global/f1", "score/global/f1_overlap_with_context"),
                    ("score/global/recall", "score/global/recall_overlap_with_context"),
                    (
                        "score/global/precision",
                        "score/global/precision_overlap_with_context",
                    ),
                ],
            ),
        ],
    )
    add_to_catalog(metric, "metrics.token_overlap_with_context", overwrite=True)

Adding a Hugginface metric
--------------------------

Unitxt provides a simple way to wrap existing Huggingface metrics without the need to write code.
This is done using the predefined :class:`HuggingfaceMetric <unitxt.metrics.HuggingfaceMetric>` class.

.. code-block:: python

    metric = HuggingfaceMetric(
        hf_metric_name="bleu",  # The name of the metric in huggingface
        main_score="bleu",      # The main score (assumes the metric returns this score name)
        prediction_type=str   # The type of the prediction and references (note that by default references are a list of the prediction_type)
    )
    add_to_catalog(metric, "metrics.bleu", overwrite=True)

By default, the HuggingfaceMetric wrapper passes only the ``prediction`` and ``references`` fields to
the metrics. You can also pass fields from the ``task_data`` inputs, by specifying ``hf_additional_input_fields``.
For example:

.. code-block:: python

    metric = HuggingfaceMetric(
        ...
        hf_additional_input_fields_pass = ["num1","num2"], # passes the task's num1 and num2 fields
        ...

    )

In the above example, ``num1`` and ``num2`` fields are passed as lists of values to the metric
(each element in the list corresponds to an instance). If you want to pass a scalar (single) value to the metric
you can use:

.. code-block:: python

    metric = HuggingfaceMetric(
        ...
        hf_additional_input_fields_pass_one_value=["tokenize"],
        ...
    )


This assumes the field has the same value is in all instances.


Note that ``Huggingface`` metrics are independent from the tasks they are used for, and receive arbitrary types of predictions, references, and additional
parameters.  A mapping may be needed between unitxt field names, values and types to the corresponding interface of the metric, using
the ``MetricPipeline`` described in the previous section.

.. note::

   Use HuggingfaceMetric to wrap metrics defined in Huggingface Hub. Do not use it to wrap Huggingface metrics implemented
   in local files.  This is because local metrics are accessed via relative or absolute file paths, and both
   may not be relevant if running code on different machines or root directories.