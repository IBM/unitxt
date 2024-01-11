import gradio as gr
from gradio_ui.load_catalog_data import load_cards_data, get_catalog_items
from unitxt.standard import StandardRecipe
import constants as cons

data = load_cards_data()
tasks = gr.Dropdown(choices=data.keys(), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[],label='Template')
instructions = gr.Dropdown(choices=[None]+get_catalog_items('instructions'), label='Instruction')
formats = gr.Dropdown(choices=[None]+get_catalog_items('formats'),label='Format')
num_shots = gr.Slider(minimum=0,maximum=5,step=1,label='Num Shots')
augmentors = gr.Dropdown(choices=[],label='Augmentor')

parameters = [tasks,cards,templates,instructions, formats, num_shots, augmentors]
# where is summarization?

#TODO - cache issue of remembering previous choice
#TODO - minimize code to copy
#TODO - move between samples
# color parts of the prompt
#TODO fix code syntax - dataset = load_dataset('unitxt/data', 'card=cards.piqa,template=templates.qa.multiple_choice.no_intro.mmlu,num_demos=2,demos_pool_size=100,instruction=instructions.empty,format=formats.user_agent')
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

    prompt = build_prompt(prompt_args)
    command = build_command(prompt_args)
    print(prompt)
    return prompt['source'],prompt['metrics'],prompt['target'],command


def build_prompt(prompt_args): 
    recipe = StandardRecipe(**prompt_args)
    dataset = recipe()
    for instance in dataset["train"]:
        return instance

def build_command(prompt_data):
    parameters_str = [f"{key}='{prompt_data[key]}'" for key in prompt_data]
    parameters_str = ",".join(parameters_str)
        
    command = f"""
    from datasets import load_dataset 

    dataset = load_dataset(
        'unitxt/data', {parameters_str}
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
     return gr.update(choices=data[task_choice].keys())

def get_augmentors(task_choice):
    print(data[task_choice][cons.AUGMENTABLE])
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
    tasks.change(update_choices_per_task,inputs=tasks,outputs=[cards,augmentors])
    cards.change(get_templates,inputs=[tasks,cards], outputs=templates)
    prompt = gr.Interface(
                fn=run_unitxt,
                inputs=parameters,
                outputs=[gr.Textbox(lines=5,show_copy_button=True,label='Prompt'),
                         gr.Textbox(lines=1,show_copy_button=True,label='Metrics'),
                         gr.Textbox(lines=1,show_copy_button=True,label='Target'),
                         gr.Textbox(lines=3,show_copy_button=True,label='Code')],
                allow_flagging=False,
                live=True,
                title="Unitxt Demo"
    )
   
    
  


demo.launch(debug=True)
