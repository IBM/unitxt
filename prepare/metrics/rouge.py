

from src.unitxt.metrics import MetricPipeline, HuggingfaceMetric
from src.unitxt.test_utils.metrics import test_metric
from src.unitxt import add_to_catalog
from src.unitxt.blocks import CastFields, CopyPasteFields
import numpy as np

metric = MetricPipeline(
    main_score='rougeL',
    preprocess_steps=[
        # CopyPasteFields(mapping=[('references/0', 'references')], use_dpath=True),
        # CastFields(
        #     fields={'prediction': 'float', 'references': 'float'}, 
        #     failure_defaults={'prediction': 0.0}, 
        #     use_dpath=True,
        # ),
    ],
    metric=HuggingfaceMetric(
        metric_name='rouge',
        main_score='rougeL',
    )
)

predictions = ["hello there", "general kenobi"]
references = [["hello", "there"], ["general kenobi", "general yoda"]]

instance_targets = [{'rouge1': 0.67, 'rouge2': 0.0, 'rougeL': 0.67, 'rougeLsum': 0.67,'score': 0.67},
                    {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0,'score': 1.0}]

global_target = {'rouge1': 0.83, 'rouge2': 0.5, 'rougeL': 0.83, 'rougeLsum': 0.83,'score': 0.83}

outputs = test_metric(
    metric=metric, 
    predictions=predictions, 
    references=references, 
    instance_targets=instance_targets, 
    global_target=global_target
)

add_to_catalog(metric, 'metrics.rouge', overwrite=True)
