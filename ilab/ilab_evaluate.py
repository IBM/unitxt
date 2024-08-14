import yaml
from lh_eval_api import load_lh_dataset
from unitxt.blocks import TaskCard, Task
from unitxt.loaders import LoadFromDictionary
from unitxt.api import load_dataset
from typing import List
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt import evaluate, load_dataset
from unitxt.inference import OpenAiInferenceEngineParams, \
     NonBatchedInstructLabInferenceEngine
import pandas as pd
from datetime import datetime
from typing import List,Dict,Any, Tuple
from datasets import DatasetDict
from unitxt import register_local_catalog
import argparse


class EvaluateIlab:
    card:str
    template:str
    task_name:str
    host_machine:str
    yaml_file:str 
    num_test_samples:int 
    local_catalog:str

    def __init__(
            self, 
            host_machine:str, 
            card:str, 
            template:str, 
            task_name:str, 
            yaml_file:str, 
            is_trained:bool,
            local_catalog:str = None,
            num_test_samples:int = 100
            ):
        self.card = card
        self.host_machine = host_machine
        self.template = template
        self.task_name = task_name
        self.yaml_file = yaml_file
        self.local_catalog = local_catalog
        self.num_test_samples= num_test_samples
        self.is_trained = is_trained
        if self.local_catalog:
            register_local_catalog(self.local_catalog)

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
            loader_limit=self.num_test_samples,
            num_demos = num_shots,
            demos_pool_size = num_shots*4,
        )
        return dataset

    def run(self):
        trained = 'trained' if self.is_trained else 'base'
        title = f'{self.yaml_file.split("/")[-1].replace(".yaml","")}_{trained}'
        for numshot in [0,5]:
            self.load_infer_and_save(num_shots=numshot,title=title)
        pass
        # add logging
        # run yaml w metrics and w llmaaj


    def load_infer_and_save(self,num_shots:int, title:str):
        title = f"{title}_{num_shots}_shots"
        csv_path = f"ilab/ilab_results/{title}.csv"
        dataset = self.load_test_data(num_shots)
        evaluated_dataset, model_name = self.infer_from_model(dataset=dataset)
        base_run_params = {'loader_limit':str(self.loader_limit),'host':self.host_machine,'folder':'instructlab','num_shots':num_shots}
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
        run_data = {
            'owner':'', 
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
        

    
    def create_dataset_from_yaml(self, metrics_list:List[str])-> DatasetDict:
        def get_data_for_loader(yaml_file:str)-> dict:
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
                yaml_content = yaml_content.get("seed_examples", {})
            data = { "test" :(yaml_content)}
            return data

        def get_card(data,metrics_list:List[str])->TaskCard:
            card = TaskCard(
                loader=LoadFromDictionary(data=data),
                task=Task(
                    inputs={"question": "str"},
                outputs={"answer": "str"},
                prediction_type="str",
                metrics=metrics_list
                ),
            templates=TemplatesDict(
                {
                    'basic': InputOutputTemplate(
                        instruction = '',
                        input_format="{question}",
                        output_format="{answer}",                
                    )
                }
            )
        )
            return card

        data = get_data_for_loader(yaml_file=self.yaml_file)
        card = get_card(data, metrics_list=metrics_list)
        dataset = load_dataset(card=card, template_card_index="basic" )
        return dataset


def example():
    host_machine = 'cccxc450'
    yaml_file = "watson-emotion_text_first.yaml"
    task_name = 'classification'
    template = "templates.classification.multi_class.text_before_instruction_with_type_of_class_i_think"
    local_catalog =  "../fm-eval/fm_eval/catalogs/private"
    card = "cards.watson_emotion"
    evaluator = EvaluateIlab(host_machine=host_machine, card=card, template=template,
                             task_name=task_name, yaml_file=yaml_file, local_catalog=local_catalog)
    evaluator.run()


if __name__ == '__main__':
    # TODO: have the loaded records exclude samples used for augmentation?
    # TODO: calc score on selected 5 samples [construct dataset from yaml?]
    # TODO: evaluation of selected 5 samples with llmaaj
    # TODO: corr of llmaaj to overall score with gold on 100 samples

   
    parser = argparse.ArgumentParser(description='evaluate dataset against ilab model and save results')
    
    parser.add_argument('--card', type=str, required=True, help='Card name')
    parser.add_argument('--template', type=str, required=True, help='Template name')
    parser.add_argument('--task_name',required=True, type=str,help='Task name, e.g. classification, translation etc.')
    parser.add_argument('--host_machine', type=str, required=True, help='Name of the host machine serving the model (e.g. cccxc450)')
    parser.add_argument('--is_trained',type=bool, required=True, help='Mark if evaluation is on trained model (TRUE) or base model (FALSE)')
    parser.add_argument('--yaml_file', type=str, required=True, help='Path of yaml file containing examples')
    parser.add_argument('--local_catalog', type=str, default=None, help='Optional: If using a non unitxt card, local Catalog path, None by default')    
    parser.add_argument('--num_test_samples', type=int, default=100, help='Optional: Num of assessed records, 100 by default')
    args = parser.parse_args()

    evaluator = EvaluateIlab(
        card = args.card,
        template = args.template,
        task_name=args.task_name,
        host_machine=args.host_machine,
        yaml_file=args.yaml_file,
        is_trained=args.is_trained,
        num_test_samples=args.num_test_samples,
        local_catalog=args.local_catalog
    )
    evaluator.run()
        