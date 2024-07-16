from dataclasses import dataclass
from typing import List, Optional
import yaml
import json
import random

from unitxt.api import load_dataset

@dataclass
class SeedExample:
    """
    Represents an example seed item with question, answer, and optionally context.
    
    Attributes:
        question (str): A question for the model
        answer (str): The desired response from the model
        context (Optional[str]): For grounded skills - context containing information that the model is expected to take into account during processing
    """
    question: str
    answer: str
    context: Optional[str] = None
    max_length: int = 2300

    def get_length(self):
        q_len = len(self.question.split())
        a_len = len(self.answer.split())
        return a_len+q_len
    
    def __post_init__(self):
        length = self.get_length()
        if length > self.max_length:
            raise ValueError(f"Question + Answer must not exceed {self.max_length} words. Currently there are ~{length} words")

    def _to_dict(self)->dict:
        data = {
                'question': self.question,
                'answer': self.answer
            }
        if self.context is not None:
                data['context'] = self.context
            
        return data

@dataclass
class IlabSkillAdder:
    """
    Represents the task description including the version, creator, and a list of seed examples.
    
    Attributes:
        task_description (str): A description of the skill.
        created_by (str): The GitHub username of the contributor.
        seed_examples (List[SeedExample]): A list of seed examples related to the skill. The file must contain 5 examples.
    """
    task_description: str
    created_by: str
    yaml_file_path: str
    seed_examples: List[SeedExample] 
    version: int = 2
    num_required_examples:int = 5

    def __post_init__(self):
        num_examples = len(self.seed_examples)
        if num_examples!= self.num_required_examples:
            raise ValueError(f"Skill Adder must contain exactly {self.num_required_examples} examples. Currently there are {num_examples}")
        self._save_to_yaml()


        
    def _save_to_yaml(self) -> None:
        def quoted_presenter(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

        yaml.add_representer(str, quoted_presenter)
        yaml.add_representer(int, yaml.representer.SafeRepresenter.represent_int)
        
        data = {
            'version': self.version,
            'task_description': self.task_description,
            'created_by': self.created_by,
            'seed_examples': [example._to_dict() for example in self.seed_examples]
        }
        
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        print(f"Data saved to {self.yaml_file_path}")



if __name__ == "__main__":
    dataset = load_dataset(card='cards.squad',template='templates.qa.with_context.simple')
    dataset =dataset['train']
    task_description = "SQUAD with context simple"
    creator = 'RF'
    yaml_file = "squad_samples.yaml"
    examples = []
    random_indexes = random.sample(range(len(dataset)), 5)
    for idx in random_indexes:
        example_data =  json.loads(dataset[idx]['task_data'])
        question = example_data['question']
        answer = example_data['answers'][0]
        context = example_data['context'] if 'context' in example_data else None
        examples.append(SeedExample(
            question=question, answer=answer, context=context
        ))

    print(f"Using the following indexes: {random_indexes}")

    IlabSkillAdder(
        task_description=task_description,
        created_by=creator,
        seed_examples=examples,
        yaml_file_path=yaml_file
    )


