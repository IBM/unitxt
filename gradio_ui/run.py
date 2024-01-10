import gradio as gr
from gradio_ui.load_catalog_data import load_cards_data, get_catalog_items
from unitxt.standard import StandardRecipe

data = load_cards_data()
tasks = gr.Dropdown(choices=data.keys(), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[],label='Template')
formats = gr.Dropdown(choices=get_catalog_items('formats'),label='Format')

#TODO - allow changing to None in each dropdown
#TODO - move between samples
#TODO - fix display of text 
#TODO - add output, score etc.
#TODO - cache dataset instead of loading from HF
# order list:
# task
# dataset
# template
# instruction
# format
# num_demos
# additional:
# augmentor

def run_unitxt(task,dataset,template,format):
    if not isinstance(dataset,str) or not isinstance(template,str):
        return '',''

    is_format = isinstance(format,str)
    prompt_args = {'card':dataset, 'template': template}
    if is_format:
        prompt_args['format'] = format 
    prompt = build_prompt(prompt_args)
    command = build_command(prompt_args)
    return prompt,command


def build_prompt(prompt_args): 
    recipe = StandardRecipe(**prompt_args)
    dataset = recipe()
    for instance in dataset["train"]:
        return instance

def build_command(prompt_data):
    parameters_str = [f"{key}='{prompt_data[key]}'" for key in prompt_data]
    parameters_str = ",".join(parameters_str)
        
    command = f"""
    from datasets import load_dataset </br>

    dataset = load_dataset('unitxt/data', {parameters_str})
    """
    return command,command

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
                inputs=[tasks,cards,templates,formats],
                outputs=[gr.Textbox(lines=10,show_copy_button=True,label='Prompt'),
                         gr.Textbox(lines=5,show_copy_button=True,label='Code')],
                allow_flagging=False,
                live=True,
                title="Unitxt Demo"
    )
   
    
  


demo.launch(debug=True)
