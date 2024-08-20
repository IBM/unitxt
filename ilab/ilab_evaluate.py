import ast
import re

import yaml
#from lh_eval_api import load_lh_dataset
from unitxt.api import load_dataset
from typing import List
from unitxt import evaluate, load_dataset
from unitxt.inference import OpenAiInferenceEngineParams, \
     NonBatchedInstructLabInferenceEngine
import pandas as pd
from datetime import datetime
from typing import List,Dict,Any, Tuple
from datasets import DatasetDict
from unitxt import register_local_catalog
import argparse
import importlib
from dataclasses import dataclass,asdict

@dataclass
class IlabRunParams:
    file:str
    yaml_indices:List[int]
    template:str
    loader_limit:int
    num_shots:int
    base_model:bool
    is_yaml:bool

    def to_dict(self):
        return asdict(self)


class EvaluateIlab:
    host_machine:str
    card:str
    task_name:str
    yaml_file:str
    is_trained:bool
    template:str
    template_index: int
    local_catalog:str
    num_test_samples:int 
    owner:str
    llmaaj_metric:str
    eval_yaml_only:bool
    lh_predictions_namespace:bool
    

    def __init__(
            self, 
            host_machine:str, 
            card:str,
            task_name:str, 
            yaml_file:str, 
            is_trained:bool = False,
            template: str = None,
            template_index: int = None,
            local_catalog:str = None,
            num_test_samples:int = 100,
            owner:str = 'ilab',
            llmaaj_metric:List[str] = ['metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn'],
            eval_yaml_only:bool = False,
            lh_predictions_namespace:bool = False
            ):
        self.card = card
        self.host_machine = host_machine
        self.template = template
        self.template_index = template_index
        self.task_name = task_name
        self.yaml_file = yaml_file
        self.local_catalog = local_catalog
        self.num_test_samples= num_test_samples
        self.is_trained = is_trained
        self.owner = owner
        if self.local_catalog:
            register_local_catalog(self.local_catalog)
        self.llmaaj_metric = llmaaj_metric
        self.eval_yaml_only = eval_yaml_only
        self.folder = 'ilab/ilab_results'
        self.lh_predictions_namespace = lh_predictions_namespace
        self.yaml_indices = self.get_yaml_indices() #reported in run details + used for evaluation

    def infer_from_model(self,dataset:DatasetDict) -> Tuple[List[Dict[str, Any]],str]:
        test_dataset = dataset['test']
        inference_model = NonBatchedInstructLabInferenceEngine(
            parameters=OpenAiInferenceEngineParams(max_tokens=1000),
            base_url = f'http://{self.host_machine}.pok.ibm.com:9000/v1'
            )
        predictions = inference_model.infer(test_dataset)
        model_name = inference_model.model
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
        print(evaluated_dataset[0]['score']['global'])
        return evaluated_dataset, model_name

    def load_test_data(self, num_shots:int):  
        dataset = load_dataset(
            card=self.card,
            template=self.template,
            template_card_index = self.template_index,
            loader_limit=self.num_test_samples,
            num_demos = num_shots,
            demos_pool_size = num_shots*4
        )
        return dataset

    def run(self, csv_path = None):
        if not csv_path:
            trained = 'trained' if self.is_trained else 'base'
            csv_path = f'{self.folder}/{self.yaml_file.split("/")[-1].replace(".yaml","")}_{trained}.csv'
        if not self.eval_yaml_only:
            for numshot in [0,5]:
                if numshot == 5 and self.num_test_samples < 50:
                    continue
                self.test_load_infer_and_save(num_shots=numshot,file=csv_path)
        self.yaml_load_infer_and_save(csv_path)
        if self.lh_predictions_namespace:
            upload_to_lh(self.folder, self.lh_predictions_namespace)

    def yaml_load_infer_and_save(self, file):
        yaml_dataset = self.create_dataset_from_yaml()
        csv_path = file.replace('.csv','_yaml_eval.csv')
        evaluated_yaml_datset, model_name = self.infer_from_model(yaml_dataset)
        self.save_results(
            csv_path=csv_path, 
            evaluated_dataset=evaluated_yaml_datset, 
            model_name=model_name,
            run_params_dict= IlabRunParams(
                file=csv_path,yaml_indices=self.yaml_indices,
                template=self.template if self.template else self.template_index,
                base_model=is_base_model(model_name),
                is_yaml=True, loader_limit=None, num_shots=None   
            ).to_dict()
        )

    def test_load_infer_and_save(self,num_shots:int, file:str):
        csv_path = file.replace('.csv',f'_{num_shots}_shots_{self.num_test_samples}_samples.csv')
        dataset = self.load_test_data(num_shots)
        evaluated_dataset, model_name = self.infer_from_model(dataset=dataset)
        base_run_params = IlabRunParams(
            file=csv_path, yaml_indices=self.yaml_indices,
            template=self.template if self.template else self.template_index,
            loader_limit=self.num_test_samples,
            num_shots=num_shots,
            base_model=is_base_model(model_name),
            is_yaml=False,
        ).to_dict()

        self.save_results(
            csv_path=csv_path, 
            evaluated_dataset=evaluated_dataset,
            model_name= model_name,
            run_params_dict=base_run_params
            )
        
    def save_results(self, csv_path, evaluated_dataset, model_name,run_params_dict = {}):
        global_scores = evaluated_dataset[0]['score']['global']
        main_score_name = global_scores.pop('score_name')
        global_main_score = global_scores[main_score_name]
        csv_path = csv_path.replace('.csv',f'_{model_name.split("/")[-1]}.csv')
        print(f"saving to {csv_path}...")
        run_data = {
            'owner':self.owner, 
            'started_at':datetime.now(), 
            'framework':'Unitxt', 
            'benchmark':'ilab', 
            'dataset':self.card.replace('cards.',''),
            'task': self.task_name,
            'model_name': model_name,
            'score': global_main_score,
            'score_name':main_score_name,
            'all_scores':global_scores,
            'run_params':run_params_dict
            }
        pd.DataFrame([run_data]).to_csv(csv_path.replace('.csv','_run.csv'),index=False)
        predictions_data = []
        for i,item in enumerate(evaluated_dataset):
            predictions_data.append({
                'record_index':i,
                'model_input':item["task_data"]["source"],
                'references':str(item["references"]),
                'processed_model_prediction': item["processed_prediction"],
                'processed_references':str(item["processed_references"]),
                'score':item["score"]["instance"]["score"],
                'score_name':item["score"]["instance"]["score_name"],
                'data_split':"test",
            })
        pd.DataFrame(predictions_data).to_csv(csv_path.replace('.csv','_predictions.csv'),index=False)
        

    def get_yaml_indices(self):
        with open(self.yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
            pattern = r"\(indices: (\[.*?\])\)"
            match = re.search(pattern, yaml_content['task_description'])
            assert match, f"yaml description should contain the chosen indices. " \
                          f"Description: {yaml_content['task_description']}"

            yaml_indices = match.group(1)
            yaml_indices = ast.literal_eval(yaml_indices)
            return yaml_indices

    def create_dataset_from_yaml(self)-> DatasetDict:
        if self.local_catalog:
            register_local_catalog(self.local_catalog)
        if self.template is not None:
            loaded_dataset = load_dataset(card=self.card, template=self.template,
                                          loader_limit=self.num_test_samples)
        elif self.template_index is not None:
            loaded_dataset = load_dataset(card=self.card, template_card_index=self.template_index,
                                          loader_limit=self.num_test_samples)
        else:
            raise ValueError("must have either template or template card index")  # TODO error if both are not none
        #llmaaj_metric =  'metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn'
        dataset = {'test': [loaded_dataset['train'][i] for i in self.yaml_indices]}
        #for instance in dataset['test']:
        #    instance['metrics'].append(llmaaj_metric)
        return dataset


def upload_to_lh(folder, namespace):
    import glob, pandas as pd, datetime,os
    from lh_eval_api import EvaluationResultsUploader, PredictionRecord,RunRecord
    from lh_eval_api.evaluation_data_services.evaluation_data_handlers.eval_uploader.evaluation_results_uploader import HandleExistingRuns
    runs_files = glob.glob(os.path.join(folder,'*_run.csv'))
    if len(runs_files) == 0:
        raise ValueError("no files found")
    print(f"Uploading {len(runs_files)} runs")
    runs = []
    all_predictions = []
    for file in runs_files:
        run_df = pd.read_csv(file)
        prediction_file = file.replace('_run.csv','_predictions.csv')
        run_df['inference_platform'] = 'ilab'
        run_df['execution_env'] = 'ilab'
        run_df['started_at'] = run_df['started_at'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f'))
        for dict_str in ['all_scores', 'run_params']:
            run_df[dict_str] = run_df[dict_str].apply(lambda x: eval(x.replace("np.float64", "float").replace("nan", "float('nan')")))
        row = run_df.iloc[0]
        run_record = RunRecord(
            **{col_name: row[col_name] for col_name in RunRecord.__dataclass_fields__ if col_name in run_df.columns}
        )
        runs.append(run_record)
        predictions_df = pd.read_csv(prediction_file)
        predictions_df['run_id'] = run_record.run_id
        predictions_df['model_prediction'] = predictions_df['processed_model_prediction']
        predictions_df['score'] = predictions_df['score'].apply(float)
        predictions = predictions_df.apply(
        lambda row: PredictionRecord(
            **{col_name: row[col_name] for col_name in PredictionRecord.__dataclass_fields__ if col_name in predictions_df.columns}
        ),
        axis=1,
        ).tolist()
        all_predictions.extend(predictions)

    uploader = EvaluationResultsUploader(
        runs=runs,
        predictions=all_predictions,
        predictions_namespace=namespace,
        handle_existing=HandleExistingRuns.IGNORE
    )
    uploader.upload()

def is_base_model(model_name:str)->bool:
    return 'merlinite-7b-lab-Q4_K_M.gguf' in model_name

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='evaluate dataset against ilab model and save results')
    
    parser.add_argument('--card', type=str, help='Card name')
    parser.add_argument('--template', type=str, help='Template name')
    parser.add_argument('--task_name', type=str,help='Task name, e.g. classification, translation etc.')
    parser.add_argument('--host_machine', type=str, required=True, help='Name of the host machine serving the model (e.g. cccxc450)')
    parser.add_argument('--yaml_file', type=str, help='Path of yaml file containing examples')
    
    parser.add_argument('--trained_model_flag',action="store_true", help='Optional: Mark if evaluation is on trained model')
    parser.add_argument('--local_catalog', type=str, default=None, help='Optional: If using a non unitxt card, local Catalog path, None by default')    
    parser.add_argument('--num_test_samples', type=int, default=100, help='Optional: Num of assessed records, 100 by default')
    parser.add_argument('--owner',type=str,default='ilab',help='Optional: Name of run owner, to be saved in result files')
    parser.add_argument('--card_config', type=str,
                        help='Optional: card_config name. It should be defined at create_ilab_skill_yaml.py')
    parser.add_argument('--only_yaml_flag', action='store_true', help='Optional: ran only yaml evaluation' )
    parser.add_argument('--lh_upload_namespace',type=str, help='Optional: specify predictions namespace in order to upload to lakehouse')
    args = parser.parse_args()

    if args.card_config is not None:
        module = importlib.import_module('create_ilab_skill_yaml')
        config = getattr(module, args.card_config)
        card = config.card
        template = config.template
        template_index = config.template_index
        task_name = config.task_description
        yaml_file = config.yaml_file
    else:
        card = args.card
        template = args.template
        task_name = args.task_name
        yaml_file = args.yaml_file


    evaluator = EvaluateIlab(
        card = card,
        template = template,
        task_name=task_name,
        host_machine=args.host_machine,
        yaml_file=yaml_file,
        is_trained=args.trained_model_flag,
        num_test_samples=args.num_test_samples,
        local_catalog=args.local_catalog,
        owner = args.owner,
        eval_yaml_only = args.only_yaml_flag,
        template_index=template_index,
        lh_predictions_namespace = args.lh_upload_namespace
    )
    evaluator.run()
    
    # Example:
    # python ilab/ilab_evaluate.py --card_config watson_emotion_classes_first_example --host_machine cccxc408 --local_catalog ../fm-eval/fm_eval/catalogs/private --only_yaml_flag
    