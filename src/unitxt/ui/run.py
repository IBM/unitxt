from functools import lru_cache

import evaluate
import gradio as gr
from transformers import pipeline

from unitxt import metric_url
from unitxt.standard import StandardRecipe
from unitxt.ui import constants as cons
from unitxt.ui.load_catalog_data import get_catalog_items, load_cards_data

metric = evaluate.load(metric_url)


def safe_add(parameter, key, args):
    if isinstance(parameter, str):
        args[key] = parameter


def run_unitxt_entry(
    task,
    dataset,
    template,
    instruction=None,
    format=None,
    num_demos=0,
    augmentor=None,
    model_name=None,
    index=0,
    max_length=cons.MAX_LENGTH,
):
    if not isinstance(dataset, str) or not isinstance(template, str):
        return "", "", "", "", "", ""
    if not isinstance(instruction, str):
        instruction = None
    if not isinstance(format, str):
        format = None
    if not isinstance(num_demos, int):
        num_demos = 0
    if not isinstance(augmentor, str):
        augmentor = None
    if not isinstance(model_name, str):
        model_name = None
    if not isinstance(index, int):
        index = 0
    if not isinstance(max_length, float):
        max_length = cons.MAX_LENGTH
    return run_unitxt(
        dataset,
        template,
        instruction,
        format,
        num_demos,
        augmentor,
        model_name,
        index,
        max_length,
    )


@lru_cache
def get_prompts(dataset, template, num_demos, instruction, format, augmentor):
    prompt_args = {"card": dataset, "template": template, cons.LOADER_LIMIT_STR: 100}
    if num_demos != 0:
        prompt_args.update(
            {"num_demos": num_demos, "demos_pool_size": cons.DEMOS_POOL_SIZE}
        )
    safe_add(instruction, "instruction", prompt_args)
    safe_add(format, "format", prompt_args)
    safe_add(augmentor, "augmentor", prompt_args)

    prompts_list = build_prompt(prompt_args)
    return prompts_list, prompt_args


@lru_cache
def run_unitxt(
    dataset,
    template,
    instruction=None,
    format=None,
    num_demos=0,
    augmentor=None,
    model_name=None,
    index=0,
    max_length=cons.MAX_LENGTH,
):
    # if not isinstance(dataset, str) or not isinstance(template, str):
    #     return "", "", "", "", "", ""
    prompts_list, prompt_args = get_prompts(
        dataset, template, num_demos, instruction, format, augmentor
    )
    selected_prompt = prompts_list[index]
    command = build_command(prompt_args)
    selected_prediction = ""
    selected_result = ""
    if model_name:
        predictions = generate(
            model_name,
            [prompt[cons.PROMPT_SOURCE_STR] for prompt in prompts_list],
            int(max_length),
        )
        selected_prediction = predictions[index]
        results = metric.compute(
            predictions=predictions,
            references=prompts_list,
        )
        selected_result = results[index]["score"]
    return (
        selected_prompt[cons.PROMPT_SOURCE_STR],
        selected_prompt[cons.PROMPT_METRICS_STR],
        selected_prompt[cons.PROPT_TARGET_STR],
        command,
        selected_prediction,
        selected_result,
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
    return sorted(datasets_list)


def get_augmentors(task_choice):
    if data[task_choice][cons.AUGMENTABLE_STR]:
        return [None, *get_catalog_items("augmentors")]
    return []


def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice, str):
        return gr.update(choices=[], value=None)
    return gr.update(choices=sorted(data[task_choice][dataset_choice]))


def generate(model_name, prompts, max_length):
    def get_prediction(generator, prompt):
        output = generator(prompt, num_return_sequences=1, max_length=max_length)
        return output[0]["generated_text"]

    def strip_predictions(predictions, prompts):
        stripped_predictions = []
        for i in range(len(predictions)):
            stripped_predictions.append(predictions[i].replace(prompts[i], ""))
        return stripped_predictions

    try:
        generator = pipeline("text-generation", model=model_name)
    except Exception as e:
        return f"""
        Try a different model name
        Exception:
        {e}
        """
    if isinstance(prompts, str):
        predictions = get_prediction(generator, prompts)
    else:
        predictions = []
        for prompt in prompts:
            prediction = get_prediction(generator, prompt)
            predictions.append(prediction)
    return strip_predictions(predictions, prompts)


# LOAD DATA
data = load_cards_data()

# UI ELEMENTS
# input
tasks = gr.Dropdown(choices=sorted(data.keys()), label="Task")
cards = gr.Dropdown(choices=[], label="Dataset Card")
templates = gr.Dropdown(choices=[], label="Template")
instructions = gr.Dropdown(
    choices=[None, *get_catalog_items("instructions")], label="Instruction"
)
formats = gr.Dropdown(choices=[None, *get_catalog_items("formats")], label="Format")
num_shots = gr.Slider(minimum=0, maximum=5, step=1, label="Num Shots")
augmentors = gr.Dropdown(choices=[], label="Augmentor")
model_choice = gr.Dropdown(
    label="model",
    choices=[None, "gpt2"],
    allow_custom_value=True,
    info="A hugging face model name can also be manually typed here",
)
sample_choice = gr.Slider(
    label="Sample Selector",
    step=1,
    minimum=0,
    maximum=cons.PROMPT_SAMPLE_SIZE,
    info="change to see a different sample from the datset",
)
max_length = gr.Number(label="Max Length for prediction")
parameters = [
    tasks,
    cards,
    templates,
    instructions,
    formats,
    num_shots,
    augmentors,
    model_choice,
    sample_choice,
    max_length,
]
# output
selected_prompt = gr.Textbox(lines=5, show_copy_button=True, label="Prompt")
metrics = gr.Textbox(lines=1, label="Metrics")
target = gr.Textbox(lines=1, label="Target")
prediction = gr.Textbox(
    lines=1, label="Model prediction", value="Select a model to get a prediction"
)
results = gr.Textbox(lines=1, label="Evaluation results")
code = gr.Code(label="Code", language="python", min_width=10)
output_components = [selected_prompt, metrics, target, code, prediction, results]


######################
### UI STARTS HERE ###
######################

demo = gr.Blocks()

with demo:
    # LOGO
    logo = gr.Image(
        cons.BANNER_PATH,
        width="25%",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
    )

    # DROPDOWN OPTIONS CHANGE ACCORDING TO SELECTION
    tasks.change(update_choices_per_task, inputs=tasks, outputs=[cards, augmentors])
    cards.change(get_templates, inputs=[tasks, cards], outputs=templates)

    result = gr.Interface(
        fn=run_unitxt_entry,
        inputs=parameters,
        outputs=output_components,
        allow_flagging="never",
        # live=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
