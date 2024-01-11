import gradio as gr
from gradio_ui.load_catalog_data import load_cards_data, get_catalog_items
from unitxt.standard import StandardRecipe

data = load_cards_data()
tasks = gr.Dropdown(choices=data.keys(), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[],label='Template')
instructions = gr.Dropdown(choices=get_catalog_items('instructions'), label='Instruction')
formats = gr.Dropdown(choices=get_catalog_items('formats'),label='Format')
num_shots = gr.Slider(minimum=0,maximum=5,step=1,label='Num Shots')
augmentors = gr.Dropdown(choices=get_catalog_items('augmentors'),label='Augmentor')

parameters = [tasks,cards,templates,instructions, formats, num_shots, augmentors]


#TODO - avoid error in dataset when task is cleared
#TODO - allow changing to None in each dropdown
#TODO - move between samples
#TODO - allow choosing augmentor only when possible
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

def get_datasets(task_choice):
    return gr.update(choices=data[task_choice].keys())


def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice,str):
        return gr.update(choices = [],value = None)
    return gr.update(choices = data[task_choice][dataset_choice])

demo = gr.Blocks()



with demo:
    tasks.change(get_datasets,inputs=tasks,outputs=cards)
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
