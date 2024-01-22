from functools import lru_cache

import evaluate
import gradio as gr
from transformers import pipeline

from unitxt.standard import StandardRecipe
from unitxt.ui import constants as cons
from unitxt.ui.load_catalog_data import get_catalog_items, load_cards_data

metric = evaluate.load(cons.UNITEXT_METRIC_STR)
data, jsons = load_cards_data()


def increase_num(current_num):
    if current_num == (cons.PROMPT_SAMPLE_SIZE - 1):
        return 0
    return current_num + 1


def decrease_num(current_num):
    if current_num == 0:
        return cons.PROMPT_SAMPLE_SIZE - 1
    return current_num - 1


def safe_add(parameter, key, args):
    if isinstance(parameter, str):
        args[key] = parameter


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
        for val in ["score_name", "score", "groups_mean_score"]:
            if val in scores:
                scores.pop(val)
        rounded_scores = {key: round(value, 3) for key, value in scores.items()}
        return list(rounded_scores.items())
    except Exception:
        return cons.EMPTY_SCORES_FRAME


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
    datasets_choices = None
    template_choices = None
    augmentors_choices = None
    if isinstance(task_choice, str):
        if task_choice in data:
            datasets_choices = gr.update(choices=get_datasets(task_choice))
            augmentors_choices = gr.update(choices=get_augmentors(task_choice))
    return datasets_choices, template_choices, augmentors_choices


def get_datasets(task_choice):
    datasets_list = list(data[task_choice].keys())
    datasets_list.remove(cons.AUGMENTABLE_STR)
    return sorted(datasets_list)


def get_augmentors(task_choice):
    if data[task_choice][cons.AUGMENTABLE_STR]:
        return [None, *get_catalog_items("augmentors")[0]]
    return []


def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice, str):
        return None
    return gr.update(choices=sorted(data[task_choice][dataset_choice]))


def generate(model_name, prompts, max_new_tokens=cons.MAX_NEW_TOKENS):
    model = pipeline(model=f"google/{model_name}")
    return [
        output["generated_text"]
        for output in model(prompts, max_new_tokens=max_new_tokens)
    ]
