from functools import lru_cache

import evaluate
import gradio as gr
from transformers import pipeline

from unitxt.standard import StandardRecipe
from unitxt.ui import constants as cons
from unitxt.ui.load_catalog_data import get_catalog_items, load_cards_data

metric = evaluate.load(cons.UNITEXT_METRIC_STR)


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
    run_model=False,
    index=0,
    # max_new_tokens=cons.MAX_NEW_TOKENS,
):
    is_dataset = isinstance(dataset, str)
    is_template = isinstance(template, str)
    if not is_dataset or not is_template:
        if is_dataset:
            msg = "Please select a template"
        else:
            msg = "Please select Dataset Card and a template"

        return msg, "", "", cons.EMPTY_SCORES_FRAME, cons.EMPTY_SCORES_FRAME, ""

    if not isinstance(instruction, str):
        instruction = None
    if not isinstance(format, str):
        format = None
    if not isinstance(num_demos, int):
        num_demos = 0
    if not isinstance(augmentor, str):
        augmentor = None
    if not isinstance(index, int):
        index = 0
    # if not isinstance(max_new_tokens, float):
    #     max_new_tokens = cons.MAX_NEW_TOKENS
    return run_unitxt(
        dataset,
        template,
        instruction,
        format,
        num_demos,
        augmentor,
        run_model,
        index,
        # max_new_tokens,
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
def get_predictions_and_scores(prompts_hashable):
    prompts_list = [unhash_dict(prompt) for prompt in prompts_hashable]
    prompts_sources = [prompt[cons.PROMPT_SOURCE_STR] for prompt in prompts_list]
    predictions = generate(
        model_name=cons.FLAN_T5_BASE,
        prompts=prompts_sources,
    )
    results = metric.compute(
        predictions=predictions,
        references=prompts_list,
    )
    return predictions, results


def hash_dict(input_dict):
    return frozenset(
        (
            key,
            hash_dict(value)
            if isinstance(value, dict)
            else tuple(value)
            if isinstance(value, list)
            else value,
        )
        for key, value in input_dict.items()
    )


def unhash_dict(input_frozenset):
    return {
        key: unhash_dict(value)
        if isinstance(value, frozenset)
        else value
        if not isinstance(value, tuple)
        else list(value)
        for key, value in input_frozenset
    }


def create_dataframe(scores):
    try:
        for val in ["score_name", "score"]:
            if val in scores:
                scores.pop(val)
        rounded_scores = {key: round(value, 3) for key, value in scores.items()}
        return list(rounded_scores.items())
    except Exception:
        return cons.EMPTY_SCORES_FRAME


@lru_cache
def run_unitxt(
    dataset,
    template,
    instruction=None,
    format=None,
    num_demos=0,
    augmentor=None,
    run_model=False,
    index=0,
    # max_new_tokens=cons.MAX_NEW_TOKENS,
):
    try:
        prompts_list, prompt_args = get_prompts(
            dataset, template, num_demos, instruction, format, augmentor
        )
        selected_prompt = prompts_list[index]
        prompt_text = selected_prompt[cons.PROMPT_SOURCE_STR]
        prompt_target = selected_prompt[cons.PROPT_TARGET_STR]
        command = build_command(prompt_args, with_prediction=run_model)
    except Exception as e:
        prompt_text = f"""
    Oops... this combination didnt work! Try something else.

    Exception: {e!r}
    """
        prompt_target = ""
        command = ""
    selected_prediction = ""
    instance_result = cons.EMPTY_SCORES_FRAME
    agg_result = cons.EMPTY_SCORES_FRAME
    if run_model:
        try:
            predictions, results = get_predictions_and_scores(
                tuple(hash_dict(prompt) for prompt in prompts_list)
            )
            selected_prediction = predictions[index]
            selected_result = results[index]["score"]
            instance_result = create_dataframe(selected_result["instance"])
            agg_result = create_dataframe(selected_result["global"])
        except Exception as e:
            selected_prediction = f"""
            An exception has occured:

            {e!r}
            """

    return (
        prompt_text,
        prompt_target,
        selected_prediction,
        instance_result,
        agg_result,
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
    except (RuntimeError, KeyError):
        prompt_list = collect_prompts("test")
    return prompt_list


def build_command(prompt_data, with_prediction):
    parameters_str = [
        f"{key}='{prompt_data[key]}'"
        for key in prompt_data
        if key != cons.LOADER_LIMIT_STR
    ]
    parameters_str = ",".join(parameters_str).replace("'", "")
    load_dataset_code = f"dataset = load_dataset('unitxt/data', '{parameters_str},max_train_instances=5', split='train')"

    code = f"""
{cons.DATASET_IMPORT_STR}

{load_dataset_code}
    """
    if with_prediction:
        imports_code = f"""
{cons.PREDICTIONS_IMPORTS_STR}
{cons.DATASET_IMPORT_STR}
        """

        code = f"""
{imports_code}

{load_dataset_code}
{cons.PREDICTION_CODE_STR}
        """
    return code


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


def generate(model_name, prompts, max_new_tokens=cons.MAX_NEW_TOKENS):
    model = pipeline(model=f"google/{model_name}")
    return [
        output["generated_text"]
        for output in model(prompts, max_new_tokens=max_new_tokens)
    ]


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
num_shots = gr.Radio(choices=list(range(6)), label="Num Shots", value=0)
augmentors = gr.Dropdown(choices=[], label="Augmentor")
run_model = gr.Checkbox(value=False, label=f"Predict with {cons.FLAN_T5_BASE}")

sample_choice = gr.Radio(
    label="Select sample to view",
    choices=list(range(cons.PROMPT_SAMPLE_SIZE)),
    info="switch to see a different sample from the dataset",
    value=0,
)
# max_new_tokens = gr.Number(label="Max New Tokens", value=cons.MAX_NEW_TOKENS)
parameters = [
    tasks,
    cards,
    templates,
    instructions,
    formats,
    num_shots,
    augmentors,
    run_model,
    sample_choice,
    # max_new_tokens,
]
# output
selected_prompt = gr.Textbox(lines=3, show_copy_button=True, label="Prompt")
metrics = gr.Textbox(lines=1, label="Metrics")
target = gr.Textbox(lines=1, label="Target")
prediction = gr.Textbox(
    lines=1,
    label="Model prediction",
    value=" ",
)
instance_scores = gr.DataFrame(
    label="Instance scores",
    value=cons.EMPTY_SCORES_FRAME,
    headers=cons.SCORE_FRAME_HEADERS,
    height=150,
)
global_scores = gr.DataFrame(
    label=f"Aggregated scores for {cons.PROMPT_SAMPLE_SIZE} predictions",
    value=cons.EMPTY_SCORES_FRAME,
    headers=cons.SCORE_FRAME_HEADERS,
    height=150,
    wrap=True,
)
code = gr.Code(label="Code", language="python", lines=1)
output_components = [
    selected_prompt,
    target,
    prediction,
    instance_scores,
    global_scores,
    code,
]


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
