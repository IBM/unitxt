import traceback
from functools import lru_cache

import gradio as gr
from transformers import pipeline

from ..api import evaluate
from ..logging_utils import get_logger
from ..standard import StandardRecipe
from ..text_utils import print_dict
from . import settings as config
from .load_catalog_data import get_catalog_items, load_cards_data

logger = get_logger()
data, jsons, formats_items, system_prompts_items = load_cards_data()


def conditionally_activate_button(conditional_element, button):
    if isinstance(conditional_element, str):
        return gr.Button(interactive=True)
    return button


def increase_num(current_num):
    if current_num == (config.PROMPT_SAMPLE_SIZE - 1):
        return 0
    return current_num + 1


def decrease_num(current_num):
    if current_num == 0:
        return config.PROMPT_SAMPLE_SIZE - 1
    return current_num - 1


def safe_add(parameter, key, args):
    if isinstance(parameter, str):
        args[key] = parameter


@lru_cache
def get_prompts(dataset, template, num_demos, system_prompt, format, augmentor):
    prompt_args = {"card": dataset, "template": template, config.LOADER_LIMIT_STR: 300}
    if num_demos != 0:
        prompt_args.update(
            {"num_demos": num_demos, "demos_pool_size": config.DEMOS_POOL_SIZE}
        )
    safe_add(system_prompt, "system_prompt", prompt_args)
    safe_add(format, "format", prompt_args)
    safe_add(augmentor, "augmentor", prompt_args)

    prompts_list = build_prompt(prompt_args)
    return prompts_list, prompt_args


@lru_cache
def get_predictions_and_scores(prompts_hashable):
    prompts_list = [unhash_dict(prompt) for prompt in prompts_hashable]
    prompts_sources = [prompt[config.PROMPT_SOURCE_STR] for prompt in prompts_list]
    predictions = generate(
        model_name=config.FLAN_T5_BASE,
        prompts=prompts_sources,
    )
    results = evaluate(
        predictions=predictions,
        data=prompts_list,
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
    def try_round(value):
        try:
            return round(value, 3)
        except:
            return value

    try:
        for val in ["score_name", "score", "groups_mean_score"]:
            if val in scores:
                scores.pop(val)

        rounded_scores = {key: try_round(value) for key, value in scores.items()}
        return list(rounded_scores.items())
    except Exception:
        logger.info("An exception occurred:\n%s", traceback.format_exc())
        return config.EMPTY_SCORES_FRAME


def collect(dataset, split, n):
    results = []
    for i, instance in enumerate(dataset[split]):
        if i > n:
            break
        results.append(instance)
    return results


def build_prompt(prompt_args):
    recipe = StandardRecipe(**prompt_args)
    logger.info("loading args:")
    print_dict(prompt_args)
    dataset = recipe()
    prompt_list = []
    try:
        prompt_list = collect(dataset, "train", config.PROMPT_SAMPLE_SIZE)
    except (KeyError, RuntimeError, ValueError):
        logger.info("An exception occurred:\n%s", traceback.format_exc())
        prompt_args["demos_taken_from"] = "test"
        logger.info("trying againg with loading args:")
        print_dict(prompt_args)
        recipe = StandardRecipe(**prompt_args)
        dataset = recipe()
        prompt_list = collect(dataset, "test", config.PROMPT_SAMPLE_SIZE)
    return prompt_list


def build_command(prompt_data, with_prediction):
    parameters_str = [
        f"{key}='{prompt_data[key]}'"
        for key in prompt_data
        if key != config.LOADER_LIMIT_STR
    ]
    parameters_str = ",".join(parameters_str).replace("'", "")
    load_dataset_code = f"dataset = load_dataset('unitxt/data', '{parameters_str},max_train_instances=5', split='train', trust_remote_code=True)"

    code = f"""
{config.DATASET_IMPORT_STR}

{load_dataset_code}
    """
    if with_prediction:
        imports_code = f"""
{config.PREDICTIONS_IMPORTS_STR}
{config.DATASET_IMPORT_STR}
        """

        code = f"""
{imports_code}

{load_dataset_code}
{config.PREDICTION_CODE_STR}
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
    datasets_list.remove(config.AUGMENTABLE_STR)
    return sorted(datasets_list)


def get_augmentors(task_choice):
    if data[task_choice][config.AUGMENTABLE_STR]:
        return [None, *get_catalog_items("augmentors")[0]]
    return []


def get_templates(task_choice, dataset_choice):
    if not isinstance(dataset_choice, str):
        return None
    return gr.update(choices=sorted(data[task_choice][dataset_choice]))


def generate(model_name, prompts, max_new_tokens=config.MAX_NEW_TOKENS):
    model = pipeline(model=f"google/{model_name}")
    return [
        output["generated_text"]
        for output in model(prompts, max_new_tokens=max_new_tokens)
    ]
