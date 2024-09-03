from unitxt.inference import IbmGenAiInferenceEngine
from unitxt.api import evaluate, load_dataset
from ilab.ilab_evaluate import save_results,IlabRunParams
from ilab.create_ilab_skill_yaml import cat,clapnq,fin_qa,watson_emotion,ner, IlabParameters
from unitxt import register_local_catalog
from unitxt.system_prompts import TextualSystemPrompt
from unitxt.formats import SystemFormat
import os

def get_base_model_predictions(test_dataset, model_name):   
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_name, max_new_tokens=1000
    )
    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)
    return evaluated_dataset

def get_prompt():
    return TextualSystemPrompt(
    "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
)
def get_format(num_shots=0):
    if num_shots == 0:
        return SystemFormat(
            model_input_format='<|system|>\n{system_prompt}\n<|user|>\n{source}\n<|assistant|>\n'
)
    if num_shots > 0:
        return SystemFormat(
            demo_format="Question:\n{source}\nAnswer:\n{target_prefix}{target}\n\n",
            model_input_format="<|system|>\n{system_prompt}\n<|user|>\n\n{demos}\nQuestion:\n{source}\nAnswer:\n<|assistant|>{target_prefix}",
        )
  

def run(csv_path_to_save, config:IlabParameters,num_shots,loader_limit=100, overwrite=False):
    model_name = "ibm/granite-7b-lab"
    template = config.template if config.template else config.template_index
    card = config.card
    csv_path_to_save = f"{csv_path_to_save.replace('.csv','')}_{config.card.replace('cards.','')}_{num_shots}_shots.csv"
    if not overwrite:
        if os.path.exists(csv_path_to_save.replace('.csv','_predictions.csv')):
            return
    if config.local_catalog:
        register_local_catalog(config.local_catalog)
    load_params = {
        'card':card,
        'loader_limit':loader_limit,
        'system_prompt':get_prompt(),
        'format':get_format(num_shots=num_shots),
        'num_demos':num_shots,
        'demos_pool_size':num_shots*4
    }
    if isinstance(template,int):
        load_params['template_card_index']=template
    else:
        load_params['template']=template
    dataset = load_dataset(**load_params)
    test_dataset = dataset['test']
    evaluated_dataset = get_base_model_predictions(test_dataset,model_name=model_name)
    save_results(
        csv_path=csv_path_to_save, 
        evaluated_dataset=evaluated_dataset, 
        model_name=model_name,
        owner='Roni', 
        card=card, 
        task_name=config.task_description, 
        run_params_dict = IlabRunParams(
            file=csv_path_to_save,
            yaml_indices=[],
            template=template,
            loader_limit=loader_limit, 
            num_shots=num_shots,
            base_model=True,
            is_yaml=False).to_dict(),
            append_model_name=False
        )

if __name__ == "__main__":
    # configs = [cat,clapnq,fin_qa,watson_emotion,ner]
    configs = [watson_emotion,ner]
    for config in configs:
        for numshots in [0,5]:
            run(f'ilab/ilab_results/granite_ilab/base',config,numshots)
    
    