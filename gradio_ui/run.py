import gradio as gr
from gradio_ui.load_catalog_data import load_cards_data, get_catalog_items
from unitxt.standard import StandardRecipe
from unitxt.artifact import UnitxtArtifactNotFoundError
import constants as cons

data = load_cards_data()
tasks = gr.Dropdown(choices=data.keys(), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[],label='Template')
instructions = gr.Dropdown(choices=[None]+get_catalog_items('instructions'), label='Instruction')
formats = gr.Dropdown(choices=[None]+get_catalog_items('formats'),label='Format')
num_shots = gr.Slider(minimum=0,maximum=5,step=1,label='Num Shots')
# with gr.Accordion('Advanced'):  
augmentors = gr.Dropdown(choices=[],label='Augmentor')

parameters = [tasks,cards,templates,instructions, formats, num_shots, augmentors]
# where is summarization?
#TODO - move between samples
#TODO - cache issue of remembering previous choice
#TODO - minimize code to copy

# color parts of the prompt
#TODO - add model, output, score etc.
# if no train, choose test (add to ui?)
# augmentor as additional option

def safe_add(parameter,key, args):
    if isinstance(parameter,str):
        args[key] = parameter


def run_unitxt(task,dataset,template,instruction,format,num_demos,augmentor):
    if not isinstance(dataset,str) or not isinstance(template,str):
        return '','','',''
    prompt_args = {'card':dataset, 'template': template}
    if num_demos!=0:
        prompt_args.update({'num_demos':num_demos, 'demos_pool_size':100})
    safe_add(instruction,'instruction',prompt_args)
    safe_add(format,'format',prompt_args)
    safe_add(augmentor,'augmentor',prompt_args)

    prompt_list = build_prompt(prompt_args)
    prompt = prompt_list[0]
    command = build_command(prompt_args)
    if 'source' not in prompt:
        return prompt,prompt,prompt,command
    return prompt['source'],prompt['metrics'],prompt['target'],command


def build_prompt(prompt_args): 
    try:
        recipe = StandardRecipe(**prompt_args)
    except UnitxtArtifactNotFoundError as e:
        return ['unitxt.artifact.UnitxtArtifactNotFoundError']

    dataset = recipe()
    prompt_list = []
    for instance in dataset["train"]:
        if len(prompt_list)==10:
            return prompt_list
        prompt_list.append(instance)

def build_command(prompt_data):
    parameters_str = [f"{key}='{prompt_data[key]}'" for key in prompt_data]
    parameters_str = ",".join(parameters_str).replace("'",'')
        
    command = f"""
    from datasets import load_dataset 

    dataset = load_dataset(
        'unitxt/data', '{parameters_str}'
    )
    """
    return command

def update_choices_per_task(task_choice):
    datasets_choices = gr.update(choices=[])
    augmentors_choices = gr.update(choices=[])
    if isinstance(task_choice,str):
         if task_choice in data:
            datasets_choices = get_datasets(task_choice)
            augmentors_choices = get_augmentors(task_choice)
    return datasets_choices, augmentors_choices

def get_datasets(task_choice):
     datasets_list = list(data[task_choice].keys())
     datasets_list.remove(cons.AUGMENTABLE)
     return gr.update(choices=datasets_list)

def get_augmentors(task_choice):
    if data[task_choice][cons.AUGMENTABLE]:
       return gr.update(choices=[None]+get_catalog_items('augmentors'))
    return gr.update(choices=[])

def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice,str):
        return gr.update(choices = [],value = None)
    return gr.update(choices = data[task_choice][dataset_choice])

################################
demo = gr.Blocks()

with demo:
    logo = gr.Image('assets/banner.png', height=50, width=70,show_label=False,show_download_button=False,show_share_button=False)
    
    tasks.change(update_choices_per_task,inputs=tasks,outputs=[cards,augmentors])
    cards.change(get_templates,inputs=[tasks,cards], outputs=templates)
    prompt = gr.Interface(
                fn=run_unitxt,
                inputs=parameters,
                outputs=[gr.Textbox(lines=5,show_copy_button=True,label='Prompt'),
                         gr.Textbox(lines=1,show_copy_button=True,label='Metrics'),
                         gr.Textbox(lines=1,show_copy_button=True,label='Target'),
                         gr.Textbox(lines=3,show_copy_button=True,label='Code'),
                         ],
                allow_flagging=False,
                live=True,
    )
   
    
  


demo.launch(debug=True)
