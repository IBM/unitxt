import pandas as pd
from unitxt import evaluate, load_dataset, register_local_catalog
from unitxt.logging_utils import get_logger
from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, FilterByCondition, Rename
from unitxt.processors import ExtractMtBenchRatingJudgment
from unitxt.test_utils.card import test_card
from unitxt.inference import IbmGenAiInferenceEngine

from lh_eval_api import LakeHouseLoader 
from typing import List,Tuple
import os, ast
from ilab_evaluate import save_results
from scipy.stats import ttest_rel

logger = get_logger()
LOCAL_CATALOG = "../fm-eval/fm_eval/catalogs/private"
force_import = type(LakeHouseLoader) # lakehouseloader import is needed here

class PostEvaluate:
    def __init__(
            self, 
            preds_file:str,
            pred_column:str = 'processed_model_prediction',
            index_column:str = 'record_index',
            score_column:str = 'score',
            judging_models:dict[dict] = {
                'mistral': {
                    'model_id':"ibm-mistralai/merlinite-7b",
                    'metric':'metrics.llm_as_judge.rating.merlinite_7b_ibm_genai_template_mt_bench_single_turn',
                    'template': "templates.response_assessment.rating.mt_bench_single_turn",
                    'format': "formats.models.mistral.instruction"
                    },
                'llama' : {
                    'model_id':"meta-llama/llama-3-70b-instruct", 
                    'metric':'metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mt_bench_single_turn',
                    'template': "templates.response_assessment.rating.generic_single_turn",
                    'format': "formats.llama3_instruct"
                    },
                # prometheus
            },
            local_catalog:str = LOCAL_CATALOG,
            dataset_split:str = 'test'
            ) -> None:
        if not os.path.exists(preds_file):
            raise ValueError(f"File doesn't exist: {preds_file}")
        runs_file = preds_file.replace('predictions','run')
        if not os.path.exists(runs_file):
            raise ValueError(
                f"Run file not found. Expecting a matching file with suffix 'run': {runs_file}"
                )
        self.preds_file = preds_file
        self.run_file = runs_file
        self.pred_column = pred_column
        self.index_column = index_column
        self.score_column = score_column
        self.judging_models = judging_models
        self.local_catalog = local_catalog
        self.dataset_split = dataset_split
        logger.info("evaluator initiated")

    def get_preds_from_file(self)->Tuple[List[int],List[str]]:
        df = pd.read_csv(self.preds_file)
        predictions = df[self.pred_column].tolist()
        indices = df[self.index_column].astype(int).tolist()
        scores = df[self.score_column].astype(float).tolist()
        logger.info(f"{len(predictions)} predictions loaded")
        return indices,predictions,scores

    def get_params_from_file(self)->Tuple[str,str,str,str,str, dict]:
        df = pd.read_csv(self.run_file)
        data = df.iloc[0].to_dict()
        model = data['model_name']
        card = f"cards.{data['dataset']}"
        run_params = ast.literal_eval(data['run_params'])
        try:
            template = run_params['template']
        except KeyError:
            raise ValueError('template data missing in file')
        owner = data['owner']
        task = data['task']
        logger.info("params collected")
        return card,template, model, owner, task, run_params


    def run(self):
        loader_limit = 100
        if self.local_catalog:
            register_local_catalog(self.local_catalog)
        card,template, model, owner,task, run_params = self.get_params_from_file()
        run_params['template']=template 
        
        # indices,predictions, orig_pred_scores = self.get_preds_from_file()
        # if template.isdigit():
        #     template = int(template)
        #     full_dataset = load_dataset(
        #         card = card,
        #         template_card_index = template,
        #         loader_limit = loader_limit
        #     )
        # else:
        #     full_dataset = load_dataset(
        #         card = card,
        #         template= template,
        #         loader_limit=loader_limit
        #     )
        # selected_dataset = [full_dataset[self.dataset_split][i] for i in indices]
        for model in self.judging_models:
            logger.info(f"Judging model: {model}")
            model_run_params = run_params.copy()
            model_csv_path = self.preds_file.replace('predictions',model)
            model_run_params['file'] = model_csv_path
            evaluated_dataset = self.evaluate_meta_task(model,orig_card=card)
            save_results(
                csv_path=model_csv_path,
                evaluated_dataset=evaluated_dataset,
                model_name=self.judging_models[model]['model_id'],
                owner=owner,
                card=card,
                task_name=task,
                run_params_dict=run_params,
                append_model_name=False
            )
            
        # for metric in self.metrics:
        #     logger.info(f"Preparing {metric} evaluation")
        #     metric_dataset = selected_dataset.copy()
        #     for instance in metric_dataset:
        #         instance['metrics']=[metric] 
        #     metric_short_str = metric.split('.rating.')[-1][:30]
        #     out_csv_prefix = self.preds_file.replace('predictions',metric_short_str)
        #     evaluated_datset = evaluate(predictions=predictions,data=metric_dataset)
        #     run_params['file'] = out_csv_prefix
        #     new_pred_scores = [item['score']['instance']['score'] for item in evaluated_datset]
        #     t_statistic, p_val = self.calc_correlation(orig_pred_scores,new_pred_scores)
        #     scores = evaluated_datset[0]['score']['global']
        #     scores['t_statistic_to_orig_metric'] = t_statistic
        #     scores['p_val_to_orig_metric'] = p_val
        #     save_results(
        #         csv_path = out_csv_prefix,
        #         evaluated_dataset = evaluated_datset,
        #         model_name=model,
        #         owner=owner,
        #         card=card,
        #         task_name=task,
        #         run_params_dict=run_params,
        #         append_model_name=False
        #     )
        #     logger.info(f"saving results: {out_csv_prefix}")

       
    # def calc_correlation(self, base_pred_scores, new_pred_scores):
    #    t_statistic, p_value = ttest_rel(base_pred_scores, new_pred_scores)
    #    return t_statistic, p_value
    #    # log + add this to file + interprate already as significant or not?
    
    def evaluate_meta_task(self, model, orig_card, with_reference:bool = False):
        if with_reference:
            task = "tasks.response_assessment.rating.single_turn_with_reference"
        else:
            task = "tasks.response_assessment.rating.single_turn"
        template = self.judging_models[model]['template']
        meta_card = TaskCard(
            LoadCSV(files={'test':self.preds_file}),
            preprocess_steps=[
                Rename(
                    field_to_field={
                        "model_input": "question",
                        "score": "rating",
                        "processed_model_prediction": "answer",
                    }
                ),
               
        # Copy(field="rating/0", to_field="rating"),
        # Copy(field="answer/0", to_field="answer"),
        ],
        task = task,
        templates = [template],
        )
        logger.info('testing meta evaluation card...')
        test_card(meta_card,strict=False,loader_limit=100)
        card_name = f"{orig_card}.response_assessment.rating.single_turn_{model}_judgment"
        logger.info('adding meta evaluation card to catalog...')
        add_to_catalog(
            meta_card,
            card_name,
            overwrite=True
        )
        logger.info('loading evaluation dataset...')
        dataset = load_dataset(
            card = card_name,
            template = template,
            format = self.judging_models[model]['format'], 
        )
        model_id = self.judging_models[model]['model_id']
        logger.info(f'Inferring with {model_id}')
        inference_model = IbmGenAiInferenceEngine(model_name=model_id)
        predictions = inference_model.infer(dataset['test'])
        logger.info('Evaluating model judgments')
        evaluated_dataset = evaluate(predictions=predictions, data=dataset['test'])
        return evaluated_dataset


if __name__ == '__main__':
    files_to_post_evaluate = [
    'entities_all_train_5_shots_100_samples_ggml-model-f16-ner.gguf_predictions',
    'clapnq_base_0_shots_100_samples_predictions',
    'cat_base_5_shots_100_samples_run',
    'fin_qa_base_0_shots_100_samples_predictions',
    'watson_emotion_classes_first_base_5_shots_100_samples_predictions'
    ]
    files_to_post_evaluate = [f"ilab/ilab_results/{file}.csv" for file in files_to_post_evaluate]
    ev = PostEvaluate(
        'ilab/ilab_results/watson_emotion_classes_first_train_yaml_eval_ggml-model-f16-watson-emotion-text-last.gguf_predictions.csv',
        dataset_split='train')
    ev.run()


