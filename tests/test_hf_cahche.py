
import unittest
from src import unitxt
from datasets import load_dataset
from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    SequentialRecipe,
    MapInstanceValues,
    FormTask,
    RenderAutoFormatTemplate,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.metrics import MetricPipeline, HuggingfaceMetric
from src.unitxt.operators import AddID, CopyPasteFields, CastFields

wnli_recipe = SequentialRecipe(
                steps=[LoadHF(path='glue', name='wnli'),
                    SplitRandomMix(mix={ 'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation',}),
                    MapInstanceValues(mappers={'label': {"0": 'entailment', "1": 'not entailment'}}),
                    AddFields(fields={'choices': ['entailment', 'not entailment'],
                                      'instruction': 'classify the relationship between the two sentences from the choices.',
                                      'dataset': 'wnli'}),
                    FormTask(
                        inputs=['choices', 'instruction', 'sentence1', 'sentence2'],
                        outputs=['label'],
                        metrics=['matrics.accuracy'],
                    ),
                    RenderAutoFormatTemplate(),
                ]
            )


rte_recipe = SequentialRecipe(
        steps = [
            LoadHF(path='glue', name='rte'),
            SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation'}),
            MapInstanceValues(mappers={'label': {"0": 'entailment', "1": 'not entailment'}}),
            AddFields(fields={'choices': ['entailment', 'not entailment'], 'dataset': 'rte'}),
            FormTask(
                inputs=['choices', 'sentence1', 'sentence2'],
                outputs=['label'],
                metrics=['metrics.accuracy'],
            ),
            RenderAutoFormatTemplate(),
        ]
    )


squad_metric = MetricPipeline(
    main_score='f1',
    preprocess_steps=[
        AddID(),
        AddFields({
            'prediction_template': {'prediction_text': 'PRED', 'id': 'ID'},
            'reference_template': {'answers': {'answer_start': [-1], 'text': 'REF'}, 'id': 'ID'},
        }, use_deepcopy=True),
        CopyPasteFields(mapping=[
                ['references', 'reference_template/answers/text'],
                ['prediction', 'prediction_template/prediction_text'],
                ['id', 'prediction_template/id'],
                ['id', 'reference_template/id'],
                ['reference_template', 'references'],
                ['prediction_template', 'prediction'],
            ], use_nested_query=True),
    ],
    metric=HuggingfaceMetric(
        metric_name='squad',
        main_score='f1',
        scale=100.0,
    ),
)

spearman_metric = MetricPipeline(
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


class TestHfCache(unittest.TestCase):
    pass
    # def test_hf_cache_enabling(self):
    #     add_to_catalog(wnli_recipe, 'tmp.recipes.wnli', overwrite=True)
    #     wnli_dataset = load_dataset(unitxt.dataset_file, 'tmp.recipes.wnli')
    #     add_to_catalog(rte_recipe, 'tmp.recipes.wnli', overwrite=True)
    #     rte_dataset = load_dataset(unitxt.dataset_file, 'tmp.recipes.wnli')
    #     self.assertEqual(rte_dataset['train'][0], wnli_dataset['train'][0])
    #
    # def test_hf_dataset_cache_disabling(self):
    #     add_to_catalog(wnli_recipe, 'tmp.recipes.wnli2', overwrite=True)
    #     wnli_dataset = load_dataset(unitxt.dataset_file, 'tmp.recipes.wnli2', download_mode='force_redownload')
    #     add_to_catalog(rte_recipe, 'tmp.recipes.wnli2', overwrite=True)
    #     rte_dataset = load_dataset(unitxt.dataset_file, 'tmp.recipes.wnli2', download_mode='force_redownload')
    #     self.assertNotEqual(rte_dataset['train'][0]['source'], wnli_dataset['train'][0]['source'])
    #


if __name__ == '__main__':
    unittest.main()
