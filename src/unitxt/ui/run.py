from functools import lru_cache

import gradio as gr

from unitxt.ui import constants as cons
from unitxt.ui.ui_tiny_utils import *
from unitxt.ui.ui_utils import (
    build_command,
    create_dataframe,
    data,
    get_catalog_items,
    get_predictions_and_scores,
    get_prompts,
    get_templates,
    hash_dict,
    jsons,
    update_choices_per_task,
)


def run_unitxt_entry(
    task,
    dataset,
    template,
    instruction=None,
    format=None,
    num_demos=0,
    augmentor=None,
    index=1,
    run_model=False,
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
        index = 1
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
def run_unitxt(
    dataset,
    template,
    instruction=None,
    format=None,
    num_demos=0,
    augmentor=None,
    run_model=False,
    index=1,
    # max_new_tokens=cons.MAX_NEW_TOKENS,
):
    index = index - 1
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


def display_json(element, cards, templates, instructions, formats):
    def get_json(element_key, value):
        if isinstance(value, str):
            if value in jsons:
                to_print = jsons[value]
                if "loader" in to_print:
                    if "path" in to_print["loader"]:
                        to_print["loader"][" path"] = to_print["loader"].pop("path")
                return to_print

            else:
                return {"": f"Error: {value}'s json not found"}
        else:
            return {"": f"Select {element_key} to view json"}

    if element == cons.DATASET_CARD:
        element_name = cards
    elif element == cons.TEMPLATE:
        element_name = templates
    elif element == cons.FORMAT:
        element_name = formats
    elif element == cons.INSTRUCTION:
        element_name = instructions
    message = get_json(element, element_name)
    return element_name, message


def display_json_button(element):
    tabs = change_tab()
    if element in jsons:
        json_el = jsons[element]
        if "loader" in json_el:
            if "path" in json_el["loader"]:
                json_el["loader"][" path"] = json_el["loader"].pop("path")
        return tabs, element, json_el

    else:
        return tabs, element, {"": f"Error: {element}'s json not found"}


######################
### UI STARTS HERE ###
######################

demo = gr.Blocks()

with demo:
    with gr.Row():
        # LOGO
        logo = gr.Image(
            cons.BANNER_PATH,
            width="10%",
            show_label=False,
            show_download_button=False,
            show_share_button=False,
            scale=0.5,
        )
        links = gr.Markdown(value=cons.INTRO_TXT)

    # input
    with gr.Row():
        with gr.Column(scale=1):
            tasks = gr.Dropdown(choices=sorted(data.keys()), label="Task", scale=3)
            with gr.Row():
                cards = gr.Dropdown(choices=[], label="Dataset Card", scale=5)
                cards_js_button = gr.Button(
                    "Dataset Card json",
                    scale=1,
                    size="sm",
                    min_width=1,
                    interactive=True,
                )
            with gr.Row():
                templates = gr.Dropdown(choices=[], label="Template", scale=5)
                templates_js_button = gr.Button(
                    "Template json", scale=1, size="sm", min_width=1, interactive=False
                )
            with gr.Row():
                instructions = gr.Dropdown(
                    choices=[None, *get_catalog_items("instructions")[0]],
                    label="Instruction",
                    scale=5,
                )
                instructiosn_js_button = gr.Button(
                    "Instruction json",
                    scale=1,
                    size="sm",
                    min_width=1,
                    interactive=False,
                )
            with gr.Row():
                formats = gr.Dropdown(
                    choices=[None, *get_catalog_items("formats")[0]],
                    label="Format",
                    scale=5,
                )
                formats_js_button = gr.Button(
                    "Format json", scale=1, size="sm", min_width=1, interactive=False
                )

            num_shots = gr.Slider(label="Num Shots", maximum=5, step=1, value=0)
            with gr.Accordion(label="Additional Parameters", open=False):
                augmentors = gr.Dropdown(choices=[], label="Augmentor")

            generat_prompts = gr.Button(value="Generate Prompts")
            model_button = gr.Button(
                value=f"Infer with {cons.FLAN_T5_BASE}", interactive=False
            )
            clear_fields = gr.ClearButton()

        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.TabItem("Prompts and predictions", id=0):
                    gr.Markdown("## Prompts:")
                    with gr.Group():
                        with gr.Row():
                            sample_choice = gr.Radio(
                                label="Browse between samples:",
                                choices=list(range(1, cons.PROMPT_SAMPLE_SIZE + 1)),
                                value=1,
                                scale=1,
                            )
                        selected_prompt = gr.Textbox(
                            lines=5, show_copy_button=True, label="Prompt"
                        )

                        target = gr.Textbox(lines=1, label="Target", scale=3)

                    gr.Markdown("## Inference:")

                    with gr.Group():
                        prediction = gr.Textbox(
                            lines=1,
                            label="Model prediction",
                            value=" ",
                        )
                        with gr.Row():
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
                with gr.TabItem("Catlog Json", id=1):
                    select_element_json = gr.Radio(
                        choices=[
                            cons.DATASET_CARD,
                            cons.TEMPLATE,
                            cons.INSTRUCTION,
                            cons.FORMAT,
                        ]
                    )
                    element_name = gr.Text(label="Displayed Json")
                    json_viewer = gr.Json(value={"Message": "Select element to watch"})

    with gr.Accordion(label="Code to reproduce above run", open=False):
        code = gr.Code(language="python", lines=1)

    tasks.change(update_choices_per_task, inputs=tasks, outputs=[cards, augmentors])
    cards.change(get_templates, inputs=[tasks, cards], outputs=templates)
    select_element_json.select(
        display_json,
        [select_element_json, cards, templates, instructions, formats],
        [element_name, json_viewer],
    )
    run_model = gr.Checkbox(value=False, visible=False)

    cards_js_button.click(display_json_button, cards, [tabs, element_name, json_viewer])

    parameters = [
        tasks,
        cards,
        templates,
        instructions,
        formats,
        num_shots,
        augmentors,
        sample_choice,
        run_model,
    ]

    outputs = {
        selected_prompt,
        target,
        prediction,
        instance_scores,
        global_scores,
        code,
    }

    clear_fields.click(lambda: [None] * len(parameters), outputs=parameters).then(
        run_unitxt_entry, parameters, outputs=outputs
    )
    generat_prompts.click(run_unitxt_entry, parameters, outputs=outputs).then(
        activate_button, outputs=model_button
    )
    model_button.click(select_checkbox, outputs=run_model).then(
        run_unitxt_entry, parameters, outputs=outputs
    ).then(deactivate_button, outputs=model_button)
    sample_choice.change(run_unitxt_entry, parameters, outputs=outputs)


if __name__ == "__main__":
    demo.launch(debug=True)
