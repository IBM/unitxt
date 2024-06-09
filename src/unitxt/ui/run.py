import traceback
from functools import lru_cache

import gradio as gr

from ..logging_utils import get_logger
from . import settings as config
from .gradio_utils import (
    activate_button,
    deactivate_button,
    go_to_intro_tab,
    go_to_json_tab,
    go_to_main_tab,
    make_group_invisible,
    make_group_visible,
    make_json_visible,
    make_mrk_down_invisible,
    make_mrk_down_visible,
    make_txt_visible,
    select_checkbox,
)
from .ui_utils import (
    build_command,
    conditionally_activate_button,
    create_dataframe,
    data,
    decrease_num,
    formats_items,
    get_predictions_and_scores,
    get_prompts,
    get_templates,
    hash_dict,
    increase_num,
    jsons,
    system_prompts_items,
    update_choices_per_task,
)

logger = get_logger()


def run_unitxt_entry(
    task,
    dataset,
    template,
    system_prompt=None,
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

        return msg, "", "", config.EMPTY_SCORES_FRAME, config.EMPTY_SCORES_FRAME, ""

    if not isinstance(system_prompt, str):
        system_prompt = None
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
        system_prompt,
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
    system_prompt=None,
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
            dataset, template, num_demos, system_prompt, format, augmentor
        )
        selected_prompt = prompts_list[index]
        prompt_text = selected_prompt[config.PROMPT_SOURCE_STR]
        prompt_target = selected_prompt[config.PROMPT_TARGET_STR]
        command = build_command(prompt_args, with_prediction=run_model)
    except Exception as e:
        logger.info("An exception occurred:\n%s", traceback.format_exc())
        prompt_text = f"""
    Oops... this combination didn't work! Try something else.

    Exception: {e!r}
    """
        prompt_target = ""
        command = ""
    selected_prediction = ""
    instance_result = config.EMPTY_SCORES_FRAME
    agg_result = config.EMPTY_SCORES_FRAME
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
            An exception has occurred:

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


def display_json_button(element: str):
    if isinstance(element, str):
        if element in jsons:
            json_el = jsons[element]
            if "loader" in json_el:
                if "path" in json_el["loader"]:
                    json_el["loader"][" path"] = json_el["loader"].pop("path")
            return (
                go_to_json_tab(),
                make_mrk_down_invisible(),
                make_txt_visible(element),
                make_json_visible(json_el),
            )

        return tabs, f"Error: {element}'s json not found", None, None
    return tabs, None, None, None


######################
### UI STARTS HERE ###
######################
demo = gr.Blocks()

with demo:
    if config.HEADER_VISIBLE:
        with gr.Row():
            links = gr.Markdown(value=config.INTRO_TXT)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group() as buttons_group:
                tasks = gr.Dropdown(choices=sorted(data.keys()), label="Task", scale=3)
                tasks_js_button = gr.Button(
                    config.JSON_BUTTON_TXT,
                    scale=1,
                    size="sm",
                    min_width=0.1,
                    interactive=False,
                    variant="secondary",
                )
                cards = gr.Dropdown(choices=[], label="Dataset Card", scale=9)
                cards_js_button = gr.Button(
                    config.JSON_BUTTON_TXT,
                    scale=1,
                    size="sm",
                    min_width=0.1,
                    interactive=False,
                    variant="secondary",
                )

                templates = gr.Dropdown(choices=[], label="Template", scale=9)
                templates_js_button = gr.Button(
                    config.JSON_BUTTON_TXT,
                    scale=1,
                    size="sm",
                    min_width=0.1,
                    interactive=False,
                    variant="secondary",
                )

                system_prompts = gr.Dropdown(
                    choices=[None, *system_prompts_items],
                    label="System Prompt",
                    scale=5,
                )
                system_prompts_js_button = gr.Button(
                    config.JSON_BUTTON_TXT,
                    scale=1,
                    size="sm",
                    min_width=1,
                    interactive=False,
                )
                formats = gr.Dropdown(
                    choices=[None, *formats_items],
                    label="Format",
                    scale=5,
                )
                formats_js_button = gr.Button(
                    config.JSON_BUTTON_TXT,
                    scale=1,
                    size="sm",
                    min_width=1,
                    interactive=False,
                )

                num_shots = gr.Slider(label="Num Shots", maximum=5, step=1, value=0)
                with gr.Accordion(label="Additional Parameters", open=False):
                    augmentors = gr.Dropdown(choices=[], label="Augmentor")

            generate_prompts_button = gr.Button(
                value="Generate Prompts", interactive=False
            )
            infer_button = gr.Button(
                value=f"Infer with {config.FLAN_T5_BASE}", interactive=False
            )
            clear_fields = gr.ClearButton()

        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                with gr.TabItem("Intro", id="intro"):
                    logo = gr.Image(
                        config.BANNER_PATH,
                        show_label=False,
                        show_download_button=False,
                        show_share_button=False,
                        container=False,
                        width="50%",
                    )
                    main_intro = gr.Markdown(config.MAIN_INTRO_TXT)
                with gr.TabItem("Demo", id="demo"):
                    with gr.Row():
                        previous_sample = gr.Button(
                            "Previous Sample", interactive=False
                        )
                        next_sample = gr.Button("Next Sample", interactive=False)
                    with gr.Group() as prompt_group:
                        prompts_title = gr.Markdown(" ## &ensp; Prompt:")
                        selected_prompt = gr.Textbox(
                            lines=5,
                            show_copy_button=True,
                            label="Prompt",
                            autoscroll=False,
                        )

                        target = gr.Textbox(lines=1, label="Target", scale=3)

                    with gr.Group(visible=False) as infer_group:
                        infer_title = gr.Markdown("## &ensp; Inference:")

                        prediction = gr.Textbox(
                            lines=5,
                            label="Model prediction",
                            value=" ",
                            autoscroll=False,
                        )

                        instance_scores = gr.DataFrame(
                            label="Instance scores",
                            value=config.EMPTY_SCORES_FRAME,
                            headers=config.SCORE_FRAME_HEADERS,
                        )
                        with gr.Accordion(
                            label=f"Aggregated scores for {config.PROMPT_SAMPLE_SIZE} predictions",
                            open=False,
                        ):
                            global_scores = gr.DataFrame(
                                value=config.EMPTY_SCORES_FRAME,
                                headers=config.SCORE_FRAME_HEADERS,
                                visible=True,
                            )
                with gr.TabItem("Code", id="code"):
                    code_intro = gr.Markdown(value=config.CODE_INTRO_TXT)
                    code = gr.Code(language="python", lines=1)
                with gr.TabItem("View Catalog", id="json"):
                    json_intro = gr.Markdown(value=config.JSON_INTRO_TXT)
                    element_name = gr.Text(label="Selected Item:", visible=False)
                    json_viewer = gr.Json(value=None, visible=False)

    if config.HEADER_VISIBLE:
        acknowledgement = gr.Markdown(config.ACK_TEXT)
    # INVISIBLE ELEMENTS FOR VALUE STORAGE
    run_model = gr.Checkbox(value=False, visible=False)
    sample_choice = gr.Number(value=0, visible=False)

    # DROPDOWNS AND JSON BUTTONS LOGIC
    tasks.select(
        update_choices_per_task, inputs=tasks, outputs=[cards, templates, augmentors]
    ).then(activate_button, outputs=tasks_js_button).then(
        deactivate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=infer_button)
    cards.select(get_templates, inputs=[tasks, cards], outputs=templates).then(
        activate_button, outputs=cards_js_button
    ).then(
        conditionally_activate_button,
        inputs=[templates, generate_prompts_button],
        outputs=generate_prompts_button,
    )

    tasks_js_button.click(
        display_json_button, tasks, [tabs, json_intro, element_name, json_viewer]
    )
    cards_js_button.click(
        display_json_button, cards, [tabs, json_intro, element_name, json_viewer]
    )
    templates.select(activate_button, outputs=templates_js_button).then(
        activate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=infer_button)
    templates_js_button.click(
        display_json_button, templates, [tabs, json_intro, element_name, json_viewer]
    )
    system_prompts.select(activate_button, outputs=system_prompts_js_button).then(
        activate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=infer_button)
    system_prompts_js_button.click(
        display_json_button,
        system_prompts,
        [tabs, json_intro, element_name, json_viewer],
    )
    formats.select(activate_button, outputs=formats_js_button).then(
        activate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=infer_button)
    formats_js_button.click(
        display_json_button, formats, [tabs, json_intro, element_name, json_viewer]
    )
    num_shots.change(activate_button, outputs=generate_prompts_button).then(
        deactivate_button, outputs=infer_button
    )
    augmentors.select(activate_button, outputs=generate_prompts_button).then(
        deactivate_button, outputs=infer_button
    )

    parameters = [
        tasks,
        cards,
        templates,
        system_prompts,
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

    # CLEAR BUTTON LOGIC
    clear_fields.click(lambda: [None] * len(parameters), outputs=parameters).then(
        run_unitxt_entry, parameters, outputs=outputs
    ).then(make_mrk_down_visible, outputs=code_intro).then(
        go_to_intro_tab, outputs=tabs
    ).then(make_group_invisible, outputs=infer_group).then(
        deactivate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=infer_button).then(
        deactivate_button, outputs=previous_sample
    ).then(deactivate_button, outputs=next_sample).then(
        deactivate_button, outputs=cards_js_button
    ).then(deactivate_button, outputs=templates_js_button).then(
        deactivate_button, outputs=formats_js_button
    ).then(deactivate_button, outputs=system_prompts_js_button)

    # GENERATE PROMPT BUTTON LOGIC
    generate_prompts_button.click(
        deactivate_button, outputs=generate_prompts_button
    ).then(deactivate_button, outputs=previous_sample).then(
        deactivate_button, outputs=next_sample
    ).then(make_group_invisible, outputs=infer_group).then(
        go_to_main_tab, outputs=tabs
    ).then(make_mrk_down_invisible, outputs=code_intro).then(
        run_unitxt_entry, parameters, outputs=outputs
    ).then(activate_button, outputs=infer_button).then(
        activate_button, outputs=previous_sample
    ).then(activate_button, outputs=next_sample)

    # INFER BUTTON LOGIC
    infer_button.click(make_group_visible, outputs=infer_group).then(
        deactivate_button, outputs=infer_button
    ).then(go_to_main_tab, outputs=tabs).then(
        make_mrk_down_invisible, outputs=code_intro
    ).then(select_checkbox, outputs=run_model).then(
        run_unitxt_entry, parameters, outputs=outputs
    )

    # SAMPLE CHOICE LOGIC
    next_sample.click(increase_num, sample_choice, sample_choice)
    previous_sample.click(decrease_num, sample_choice, sample_choice)
    sample_choice.change(run_unitxt_entry, parameters, outputs=outputs)

if __name__ == "__main__":
    demo.launch(debug=True)
