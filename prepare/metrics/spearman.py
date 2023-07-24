

from src.unitxt.metrics import HFSpearman, MetricPipeline, HuggingfaceMetric
from src.unitxt.test_utils.metrics import test_metric
from src.unitxt import add_to_catalog
from src.unitxt.blocks import CastFields, CopyPasteFields
import numpy as np

metric = MetricPipeline(
    main_score='spearmanr',
    preprocess_steps=[
        CopyPasteFields(mapping=[('references/0', 'references')], use_nested_query=True),
        CastFields(
            fields={'prediction': 'float', 'references': 'float'}, 
            failure_defaults={'prediction': 0.0}, 
            use_nested_query=True,
        ),
    ],
    metric=HuggingfaceMetric(
        metric_name='spearmanr',
        main_score='spearmanr',
    )
)

predictions = ['1.0',' 2.0', '1.0']
references = [['-1.0'], ['1.0'], ['0.0']]

instance_targets = [{'spearmanr': np.nan, 'score': np.nan},
                    {'spearmanr': np.nan, 'score': np.nan},
                    {'spearmanr': np.nan, 'score': np.nan}]

global_target = {'spearmanr': 0.87, 'score': 0.87}

outputs = test_metric(
    metric=metric, 
    predictions=predictions, 
    references=references, 
    instance_targets=instance_targets, 
    global_target=global_target
)

add_to_catalog(metric, 'metrics.spearman', overwrite=True)
