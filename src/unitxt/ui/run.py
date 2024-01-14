import gradio as gr

from ..standard import StandardRecipe
from . import constants as cons
from .load_catalog_data import get_catalog_items, load_cards_data

# TODO - move between samples
# TODO - minimize code to copy, minimize augmentor
# TODO additional comments from Yotam
# TODO color parts of the prompt
# TODO - add model, output, score etc.


def safe_add(parameter, key, args):
    if isinstance(parameter, str):
        args[key] = parameter


def run_unitxt(task, dataset, template, instruction, format, num_demos, augmentor):
    if not isinstance(dataset, str) or not isinstance(template, str):
        return "", "", "", ""
    prompt_args = {"card": dataset, "template": template, cons.LOADER_LIMIT_STR: 100}
    if num_demos != 0:
        prompt_args.update(
            {"num_demos": num_demos, "demos_pool_size": cons.DEMOS_POOL_SIZE}
        )
    safe_add(instruction, "instruction", prompt_args)
    safe_add(format, "format", prompt_args)
    safe_add(augmentor, "augmentor", prompt_args)

    prompts_list = build_prompt(prompt_args)
    prompt = prompts_list[0]
    command = build_command(prompt_args)
    return (
        prompt[cons.PROMPT_SOURCE_STR],
        prompt[cons.PROMPT_METRICS_STR],
        prompt[cons.PROPT_TARGET_STR],
        command,
    )


def build_prompt(prompt_args):
    def collect_prompts(split_name):
        prompt_list = []
        for instance in dataset[split_name]:
            if len(prompt_list) == cons.PROMPT_SAMPLE_SIZE:
                return prompt_list
            prompt_list.append(instance)
        return None

    recipe = StandardRecipe(**prompt_args)
    dataset = recipe()
    prompt_list = []
    try:
        prompt_list = collect_prompts("train")
    except RuntimeError:
        prompt_list = collect_prompts("test")
    return prompt_list


def build_command(prompt_data):
    parameters_str = [
        f"{key}='{prompt_data[key]}'"
        for key in prompt_data
        if key != cons.LOADER_LIMIT_STR
    ]
    parameters_str = ",".join(parameters_str).replace("'", "")

    return f"dataset = load_dataset('unitxt/data', '{parameters_str}')"


def update_choices_per_task(task_choice):
    datasets_choices = gr.update(choices=[])
    augmentors_choices = gr.update(choices=[])
    if isinstance(task_choice, str):
        if task_choice in data:
            datasets_choices = gr.update(choices=get_datasets(task_choice))
            augmentors_choices = gr.update(choices=get_augmentors(task_choice))
    return datasets_choices, augmentors_choices


def get_datasets(task_choice):
    datasets_list = list(data[task_choice].keys())
    datasets_list.remove(cons.AUGMENTABLE_STR)
    return datasets_list


def get_augmentors(task_choice):
    if data[task_choice][cons.AUGMENTABLE_STR]:
        return [None, *get_catalog_items("augmentors")]
    return []


def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice, str):
        return gr.update(choices=[], value=None)
    return gr.update(choices=data[task_choice][dataset_choice])


# LOAD DATA
data = load_cards_data()

# UI ELEMENTS
# input
tasks = gr.Dropdown(choices=data.keys(), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[], label="Template")
instructions = gr.Dropdown(
    choices=[None, *get_catalog_items("instructions")], label="Instruction"
)
formats = gr.Dropdown(choices=[None, *get_catalog_items("formats")], label="Format")
num_shots = gr.Slider(minimum=0, maximum=5, step=1, label="Num Shots")
augmentors = gr.Dropdown(choices=[], label="Augmentor")
parameters = [tasks, cards, templates, instructions, formats, num_shots, augmentors]
# output
prompt = gr.Textbox(lines=5, show_copy_button=True, label="Prompt")
metrics = gr.Textbox(lines=1, show_copy_button=True, label="Metrics")
target = gr.Textbox(lines=1, show_copy_button=True, label="Target")
code = gr.Code(label="Code", language="python", min_width=10)
output_components = [prompt, metrics, target, code]


######################
### UI STARTS HERE ###
######################

demo = gr.Blocks()

with demo:
    # LOGO
    logo = gr.Image(
        "assets/banner.png",
        height=50,
        width=70,
        show_label=False,
        show_download_button=False,
        show_share_button=False,
    )

    # DROPDOWN OPTIONS CHANGE ACCORDING TO SELECTION
    tasks.change(update_choices_per_task, inputs=tasks, outputs=[cards, augmentors])
    cards.change(get_templates, inputs=[tasks, cards], outputs=templates)

    get_prompts = gr.Interface(
        fn=run_unitxt,
        inputs=parameters,
        outputs=output_components,
        allow_flagging="never",
        live=True,
    )
