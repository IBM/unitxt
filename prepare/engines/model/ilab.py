from unitxt import evaluate, load_dataset
from unitxt.inference import OpenAiInferenceEngineParams, \
     NonBatchedInstructLabInferenceEngine
import pandas as pd
from datetime import datetime
from typing import List,Dict,Any, Tuple
from datasets import DatasetDict

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

def infer_and_save(host_machine:str,dataset:DatasetDict, csv_path:str, dataset_name:str, task_name:str, run_params_dict:Dict={}):
    evaluated_dataset, model_name = infer_from_model(host_machine=host_machine,dataset=dataset)
    save_results(
        csv_path=csv_path, 
        evaluated_dataset=evaluated_dataset,
        dataset_name=dataset_name,
        task_name=task_name,
        model_name= model_name,
        run_params_dict=run_params_dict
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
    


if __name__ == '__main__':
    card = "cards.cnn_dailymail"
    status = 'cnn1_model'
    template = "templates.summarization.abstractive.instruct_full"
    save_csv = f"{card.split('.')[1]}_{status}.csv"
    host_machine = 'cccxc434'
    loader_limit = 100
    dataset = load_dataset(
        card=card,
        template=template,
        loader_limit=loader_limit,
        # "metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn],
    )
 

    infer_and_save(
        host_machine=host_machine,
        dataset=dataset,
        csv_path=save_csv,
        dataset_name=card,
        task_name=template,
        run_params_dict={'loader_limit':str(loader_limit),'host':host_machine,'folder':'instructlab'}
    )


    # TODO: have the loaded records exclude samples used for augmentation?
    # TODO: calc score on selected 5 samples [construct dataset from yaml?]
    # TODO: evaluation of selected 5 samples with llmaaj
    # TODO: corr of llmaaj to overall score with gold on 100 samples