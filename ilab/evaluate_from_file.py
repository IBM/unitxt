import pandas as pd, numpy as np
from unitxt import evaluate, load_dataset, register_local_catalog
from unitxt.logging_utils import get_logger,get_settings
from unitxt.blocks import (
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.operators import Rename,Cast,ExecuteExpression
from unitxt.inference import IbmGenAiInferenceEngine

from lh_eval_api import LakeHouseLoader 
from typing import List,Tuple
import os, ast
from dataclasses import dataclass
from ilab_evaluate import save_results

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True

LOCAL_CATALOG = "../fm-eval/fm_eval/catalogs/private"
force_import = type(LakeHouseLoader) # lakehouseloader import is needed here

@dataclass
class JudgeModelParams:
    name:str
    model_id:str
    template:str
    format:str
    ref_multiplier:int

JUDGE_MODELS = [
    JudgeModelParams(name='mixtral', model_id="mistralai/mixtral-8x7b-instruct-v01", format="formats.models.mistral.instruction", ref_multiplier=10, template="templates.response_assessment.rating.mt_bench_single_turn"),
    JudgeModelParams(name='llama', model_id="meta-llama/llama-3-70b-instruct", template= "templates.response_assessment.rating.generic_single_turn", format="formats.llama3_instruct", ref_multiplier=10),
    # JudgeModelParams(name='prometheus', model_id= 'kaist-ai/prometheus-8x7b-v2', format="formats.models.mistral.instruction", ref_multiplier=5)
    # CREATE TEMPLATE
 #https://github.com/prometheus-eval/prometheus-eval?tab=readme-ov-file#-quick-start
                        # https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0
                        #build upon mt_bench_single_turn_with_reference.py
                        #use dedicated word processor
                        #NOTE uses 5 score and not 10
]

class PostEvaluate:
    def __init__(
            self, 
            preds_file:str,
            pred_column:str = 'processed_model_prediction',
            index_column:str = 'record_index',
            score_column:str = 'score',
            judging_models:List[JudgeModelParams]= JUDGE_MODELS,
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

    def get_params_from_file(self)->Tuple[str,str,str,str,str, dict]:
        df = pd.read_csv(self.run_file)
        data = df.iloc[0].to_dict()
        model = data['model_name']
        card = f"cards.{data['dataset']}"
        print(data['run_params'])
        run_params = ast.literal_eval(data['run_params'])
        try:
            template = run_params['template']
        except KeyError:
            raise ValueError('template data missing in file')
        owner = data['owner']
        task = data['task']
        logger.info("params collected")
        return card,template, model, owner, task, run_params

    
    def run(self, overwrite = False):
        if self.local_catalog:
            register_local_catalog(self.local_catalog)
        card,template, model, owner,task, run_params = self.get_params_from_file()
        run_params['template']=f"'{template}'" 
        for modelparams in self.judging_models:
            model = modelparams.name
            for with_reference in [True,False]:
                model_csv_path = self.preds_file.replace('predictions',f"{model}{'_w_ref' if with_reference else ''}")
                if not overwrite:
                    if os.path.exists(model_csv_path.replace('.csv','_predictions.csv')):
                        logger.info(f"**** file already exists, skipping: {model_csv_path}")
                        continue
                model_id = modelparams.model_id
                logger.info(f"Judging model: {model_id}")
                model_run_params = run_params.copy()
                model_run_params['file'] = model_csv_path
                model_run_params['meta_eval']='True'
                model_run_params['with_reference'] = with_reference
                model_run_params['judge_template'] = modelparams.template
                model_run_params['judge_format'] = modelparams.format
                try:
                    evaluated_dataset = self.evaluate_meta_task(modelparams,with_reference=with_reference)
                except Exception as e:
                    logger.error(f"**** Error while inferring for: {model_csv_path}")
                    raise e
                save_results(
                    csv_path=model_csv_path,
                    evaluated_dataset=evaluated_dataset,
                    model_name=model_id,
                    owner=owner,
                    card=card,
                    task_name=task,
                    run_params_dict=model_run_params,
                    append_model_name=False
                )
            
    
    def evaluate_meta_task(self, modelparams:JudgeModelParams, with_reference:bool = False):
        task =  "tasks.response_assessment.rating.single_turn"
        template = modelparams.template
        if with_reference:
            add_str = "_with_reference"
            task = task+add_str
            template = template+add_str
        
        meta_card = TaskCard(
            LoadCSV(files={'test':self.preds_file}),
            preprocess_steps=[
                Rename(
                    field_to_field={
                        "unformatted_input": "question", 
                        # "score": "rating",
                        "processed_model_prediction": "answer",
                        "references": "reference_answer",
                    }
                ),
                ExecuteExpression(expression=f"float(score)*{modelparams.ref_multiplier}", to_field="rating"),
                Cast(to="str", failure_default='None', field_to_field={"answer":"answer"})                
        ],
        task = task,
        templates = [template],
        )
        
        logger.info('loading evaluation dataset...')
        dataset = load_dataset(
            card = meta_card,
            template = template,
            format = modelparams.format, 
        )
        model_id = modelparams.model_id
        logger.info(f'Inferring with {model_id}')
        inference_model = IbmGenAiInferenceEngine(model_name=model_id)
        predictions = inference_model.infer(dataset['test'])
        logger.info('Evaluating model judgments')
        evaluated_dataset = evaluate(predictions=predictions, data=dataset['test'])
        return evaluated_dataset


if __name__ == '__main__':
    from glob import glob
    files_to_post_evaluate = glob('ilab/ilab_results/granite_ilab/base_*_shots_predictions.csv')
    for file in files_to_post_evaluate:
        ev = PostEvaluate(
            file, dataset_split='test')
        ev.run(overwrite=False)
   