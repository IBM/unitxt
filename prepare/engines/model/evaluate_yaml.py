import yaml
from unitxt.blocks import TaskCard, Task
from unitxt.loaders import LoadFromDictionary
from unitxt.api import load_dataset
from typing import List
from unitxt.templates import InputOutputTemplate, TemplatesDict


def create_dataset_from_yaml(yaml_file:str, metrics_list:List[str]):
    data = get_data_for_loader(yaml_file=yaml_file)
    card = get_card(data, metrics_list=metrics_list)
    dataset = load_dataset(card=card, template_card_index="basic" )
    return dataset

def get_data_for_loader(yaml_file:str):
    with open(yaml_file, 'r') as f:
        yaml_content = yaml.safe_load(f)
        yaml_content = yaml_content.get("seed_examples", {})
    data = { "test" :(yaml_content)}
    return data

def get_card(data,metrics_list):
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

if __name__ == '__main__':
    dataset = create_dataset_from_yaml('sdg/watson_emotion.yaml',['metrics.rouge'])
    print(dataset['test'][0])