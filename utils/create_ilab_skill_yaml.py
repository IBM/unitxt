from lh_eval_api import load_lh_dataset
from unitxt.api import load_dataset
from unitxt import register_local_catalog
from typing import List, Optional
import yaml
import json
import random
from dataclasses import dataclass

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



def cat_example():
    register_local_catalog("../fm-eval/fm_eval/catalogs/private")
    task_description = "cat multi label"
    creator = 'RF'
    yaml_file = "cat_samples.yaml"

    template = "templates.classification.multi_label.text_before_instruction_with_type_of_classes_and_none"
    card = 'cards.cat'

    question_field = 'source'  # question
    answer_field = 'references'  # answers
    context_field = 'context'

    loaded_dataset = load_dataset(card=card, template=template)
    dataset = loaded_dataset['train']

    examples = []
    random_indexes = random.sample(range(len(dataset)), 5)
    for idx in random_indexes:
        # example_data =  json.loads(dataset[idx]['task_data'])
        example_data = dataset[idx]
        question = example_data[question_field]
        answer = example_data[answer_field]
        context = example_data[context_field] if context_field in example_data else None
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


def cnn_dailymail_example():
    dataset = load_dataset(card="cards.cnn_dailymail", template="templates.summarization.abstractive.instruct_full",
                           loader_limit=1000, )
    dataset = dataset['train']
    task_description = "dailymail summerizaion with context simple"
    creator = 'roni'
    yaml_file = "dailymail_summarization_w_context.yaml"
    examples = []
    random_indexes = random.sample(range(len(dataset)), 5)
    for idx in random_indexes:
        example_data = json.loads(dataset[idx]['task_data'])
        question = 'Summarize the following article.\n'
        answer = example_data['summary']
        context = example_data['document']
        examples.append(SeedExample(
            question=question, answer=answer, context=context
        ))

    print(f"Using the following indexes: {random_indexes}")
    task_description = f"{task_description}. Indices: {random_indexes}"
    IlabSkillAdder(
        task_description=task_description,
        created_by=creator,
        seed_examples=examples,
        yaml_file_path=yaml_file
    )

    
if __name__ == "__main__":
    cnn_dailymail_example()


