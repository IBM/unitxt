from unitxt import evaluate, load_dataset
from unitxt.inference import OpenAiInferenceEngineParams, \
     NonBatchedInstructLabInferenceEngine
import pandas as pd
from datetime import datetime
from typing import List,Dict,Any, Tuple
from datasets import DatasetDict
from unitxt import register_local_catalog
import argparse

def infer_from_model(host_machine:str,dataset:DatasetDict) -> Tuple[List[Dict[str, Any]],str]:
    test_dataset = dataset['test']
    inference_model = NonBatchedInstructLabInferenceEngine(
        parameters=OpenAiInferenceEngineParams(max_tokens=1000),
        base_url = f'http://{host_machine}.pok.ibm.com:9000/v1'
        )
    predictions = inference_model.infer(test_dataset)
    model_name = inference_model.model
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
    print(evaluated_dataset[0]['score']['global'])
    return evaluated_dataset, model_name

def load_data(card:str, template:str, loader_limit:int = 100, num_shots:int = 0, local_catalog=None):   
    from lh_eval_api import load_lh_dataset
    if local_catalog:
        register_local_catalog(local_catalog)
    dataset = load_dataset(
        card=card,
        template=template,
        loader_limit=loader_limit,
        num_demos = num_shots,
        demos_pool_size = num_shots*4,
    )
    return dataset


def load_infer_and_save(host_machine:str, title:str, card:str, 
                        task_name:str, template:str, 
                        num_shots:int = 0, loader_limit:int = 100, local_catalog:str = None,
                        run_params_dict:Dict={}):
    title = f"{title}_{num_shots}_shots"
    csv_path = f"ilab/ilab_results/{card.split('.')[1]}_{title}.csv"
    dataset = load_data(card,template, title, loader_limit, num_shots, local_catalog)
    evaluated_dataset, model_name = infer_from_model(host_machine=host_machine,dataset=dataset)
    base_run_params = {'loader_limit':str(loader_limit),'host':host_machine,'folder':'instructlab','num_shots':num_shots}
    save_results(
        csv_path=csv_path, 
        evaluated_dataset=evaluated_dataset,
        dataset_name=card.replace('cards.',''),
        task_name=task_name,
        model_name= model_name,
        run_params_dict=base_run_params.update(run_params_dict)
        )

def save_results(csv_path,evaluated_dataset, dataset_name, task_name, model_name,run_params_dict = {}):
    global_scores = evaluated_dataset[0]['score']['global']
    main_score_name = global_scores.pop('score_name')
    global_main_score = global_scores[main_score_name]
    run_data = {
        'owner':'', 
        'started_at':datetime.now(), 
        'framework':'Unitxt', 
        'benchmark':'ilab', 
        'dataset':dataset_name,
        'task': task_name,
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
    
def example():
    host_machine = 'cccxc450'
    title = "watson-emotion_text_first"
    task_name = 'classification'
    template = "templates.classification.multi_class.text_before_instruction_with_type_of_class_i_think"
    num_shots = 0
    local_catalog =  "../fm-eval/fm_eval/catalogs/private"
    card = "cards.watson_emotion"
    load_infer_and_save(host_machine=host_machine,
                        title=title,
                        card=card,
                        task_name=task_name,
                        template=template,
                        num_shots=num_shots,
                        local_catalog=local_catalog
                        )
    
if __name__ == '__main__':
    # TODO: have the loaded records exclude samples used for augmentation?
    # TODO: calc score on selected 5 samples [construct dataset from yaml?]
    # TODO: evaluation of selected 5 samples with llmaaj
    # TODO: corr of llmaaj to overall score with gold on 100 samples

   
    parser = argparse.ArgumentParser(description='evaluate dataset against ilab model and save results')
    
    parser.add_argument('--card', type=str, required=True, help='Card name')
    parser.add_argument('--template', type=str, required=True, help='Template name')
    parser.add_argument('--host_machine', type=str, required=True, help='Name of the host machine serving the model (e.g. cccxc450)')
    parser.add_argument('--experiment_title', type=str, required=True, help='Title for the experiment, to be part of the saved csv name')
    parser.add_argument('--task_name',required=True, type=str,help='Task name, e.g. classification, translation etc.')
    parser.add_argument('--loader_limit', type=int, default=100, help='Optional: Num of assessed records, 100 by default')
    parser.add_argument('--local_catalog', type=str, default=None, help='Optional: If using a non unitxt card, local Catalog path, None by default')
    parser.add_argument('--num_shots', type=int, default=0, help='Optional: Number of shots, 0 by default')
    parser.add_argument('--run_params_dict', type=str, default={}, help="Parameters to record in LH, in dict format. For example {'sdg_model':'GenAI'}")
    args = parser.parse_args()

    load_infer_and_save(host_machine=args.host_machine,title=args.title,
                        card = args.card, task_name=args.task_name,
                        num_shots=args.num_shots, loader_limit=args.loader_limit,
                        local_catalog=args.local_catalog,
                        run_params_dict=args.run_params_dict)
        