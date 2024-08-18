from lh_eval_api import load_lh_dataset
from unitxt.api import load_dataset
from unitxt import register_local_catalog
from typing import List, Optional
import yaml
from collections import Counter
import random
from dataclasses import dataclass
from unitxt.settings_utils import get_settings

settings = get_settings()
settings.allow_unverified_code = True

FM_EVAL_LOCAL_CATALOG = "../fm-eval/fm_eval/catalogs/private"
CREATOR = "RF"

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
    Represents the task description including the sdgbuilder, creator, and a list of seed examples.
    
    Attributes:
        task_description (str): A description of the skill.
        created_by (str): The GitHub username of the contributor.
        seed_examples (List[SeedExample]): A list of seed examples related to the skill. The file must contain 5 examples.
    """
    task_description: str
    created_by: str
    yaml_file_path: str
    seed_examples: List[SeedExample] 
    data_builder:str = "skills_sdg"
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
            'data_builder': self.data_builder,
            'task_description': self.task_description,
            'created_by': self.created_by,
            'seed_examples': [example._to_dict() for example in self.seed_examples]
        }
        
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        print(f"Data saved to {self.yaml_file_path}")

@dataclass
class IlabParameters:
    task_description:str
    creator:str
    yaml_file:str
    card:str
    template:str = None
    template_index:int = None
    question_field:str = 'source'
    answer_field:str = 'target'
    context_field:str = None
    loader_limit:int = 100
    local_catalog:str = None
    use_question_field_as_text:bool=False
    

def select_indices_by_classes(dataset, num_samples):
    def get_target_indices(target,dataset, num_indices):
        target_indices = [i for i,x in enumerate(dataset) if target in x['target']]
        return random.sample(target_indices,num_indices)

    indices = []
    freq_classes = Counter(dataset['target']).most_common(num_samples)
    n = len(freq_classes)
    base, remainder = divmod(num_samples,n)
    distribution = [base] * n
    for i in range(remainder):
        distribution[i]+=1

    for i,cls in enumerate(freq_classes):
        target = cls[0]
        target_num_samples = distribution[i]
        print(f"Fetching {target_num_samples} samples for target {target}")
        indices.extend(get_target_indices(target,dataset,target_num_samples))
    
    return indices

def select_random_indices(dataset, num_samples):
    return random.sample(range(len(dataset)), num_samples)

def create_yaml(parameters:IlabParameters,distribute=True):
    if parameters.local_catalog:
        register_local_catalog(parameters.local_catalog)
    if parameters.template is not None:
        loaded_dataset = load_dataset(card=parameters.card, template =parameters.template, loader_limit = parameters.loader_limit)
    elif parameters.template_index is not None:
        loaded_dataset = load_dataset(card=parameters.card, template_card_index=parameters.template_index, loader_limit = parameters.loader_limit)
    else:
        ValueError("must have either template or template card index") #TODO error if both are not none
    dataset = loaded_dataset['train']
    examples = []
    if distribute:
        indices = select_indices_by_classes(dataset,5)
    else:
        indices = select_random_indices(dataset,5)
    parameters.task_description = parameters.task_description + f" (indices: {indices})"
    for idx in indices:
        example_data = dataset[idx]
        # if 'task_data' in example_data:
        #     example_data = json.loads(example_data['task_data'])

        if parameters.use_question_field_as_text:
            question = parameters.question_field
        else:
            question = example_data[parameters.question_field]
        answer = example_data[parameters.answer_field]
        context = example_data[parameters.context_field] if parameters.context_field else None
        examples.append(SeedExample(
            question=question, answer=answer, context=context
        ))
    print(f"Using the following indices: {indices}")

    IlabSkillAdder(
            task_description=parameters.task_description,
            created_by=parameters.creator,
            seed_examples=examples,
            yaml_file_path=parameters.yaml_file
        )

cat_example = IlabParameters(
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    task_description="cat multi label",
    creator = CREATOR,
    yaml_file="cat_samples.yaml",
    template="templates.classification.multi_label.text_before_instruction_with_type_of_classes_and_none",
    card='cards.cat',
    question_field = 'source',  
    answer_field = 'references',
    context_field = 'context',
)

cnn_example = IlabParameters(
    task_description="dailymail summarization with context simple",
    card="cards.cnn_dailymail",
    creator = CREATOR,
    yaml_file="dailymail_summarization_w_context.yaml",
    template="templates.classification.multi_label.text_before_instruction_with_type_of_classes_and_none",
    question_field = 'Summarize the following article.\n',  
    use_question_field_as_text=True,
    answer_field = 'summary',
    context_field = 'document',
)

watson_emotion_text_first_example = IlabParameters(
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    task_description="watson emotion",
    loader_limit=100,
    creator=CREATOR,
    yaml_file="ilab/sdg/watson_emotion.yaml",
    card="cards.watson_emotion",
    template="templates.classification.multi_class.text_before_instruction_with_type_of_class_i_think",
   )

watson_emotion = IlabParameters(
    template="templates.classification.multi_class.text_after_instruction_with_type_of_class",
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    task_description="watson emotion",
    loader_limit=100,
    creator=CREATOR,
    yaml_file="ilab/sdg/watson_emotion_classes_first.yaml",
    card="cards.watson_emotion",
   question_field = 'source',  
    answer_field = 'target',
)

fin_qa = IlabParameters(
    template_index = 0,
    task_description='QA',
    creator=CREATOR,
    yaml_file='ilab/sdg/fin_qa.yaml',
    card='cards.fin_qa',
)

clapnq = IlabParameters(
    card='cards.rag.response_generation.clapnq',
    task_description='rag',
    creator=CREATOR,
    yaml_file='ilab/sdg/clapnq.yaml',
    template="templates.rag.response_generation.please_respond"
)

pagerduty = IlabParameters(
    card='cards.pagerduty',
    template='templates.summarization.dialogue.dialogue_here_instruction_dialogue_field_write_summary_instruction',
    task_description='summarization',
    yaml_file='ilab/sdg/pagerduty.yaml',
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    creator=CREATOR
)

persado = IlabParameters(
    card='cards.persado',
    template="templates.generation.question_generation.given_input",
    task_description='persado',
    yaml_file='ilab/sdg/persado.yaml',
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    creator=CREATOR
)

ner = IlabParameters(
    card='cards.entities_selected.all',
    template='templates.ner.all_entity_types.extract_all_entities',
    task_description='ner',
    yaml_file='ilab/sdg/entities_all.yaml',
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    creator=CREATOR
)

cat = IlabParameters(
    card='cards.cat',
    template='templates.classification.multi_label.text_before_instruction_with_type_of_classes_and_none',
    task_description='classification',
    yaml_file='ilab/sdg/cat.yaml',
    local_catalog=FM_EVAL_LOCAL_CATALOG,
    creator=CREATOR
)
    
if __name__ == "__main__":
    create_yaml(cat)


