import gradio as gr


def go_to_json_tab():
    return gr.Tabs(selected="json")


def go_to_main_tab():
    return gr.Tabs(selected="demo")


def go_to_intro_tab():
    return gr.Tabs(selected="intro")


def activate_button():
    return gr.Button(interactive=True)


def deactivate_button():
    return gr.Button(interactive=False)


def make_button_visible():
    return gr.Button(visible=True)


def make_button_invisible():
    return gr.Button(visible=False)


def select_checkbox():
    return gr.Checkbox(value=True)


def deselect_checkbox():
    return gr.Checkbox(value=False)


def make_txt_visible(value):
    return gr.Text(visible=True, value=value)


def make_json_visible(value):
    return gr.Json(visible=True, value=value)


def make_mrk_down_invisible():
    return gr.Markdown(visible=False)


def make_mrk_down_visible():
    return gr.Markdown(visible=True)


def make_group_visible():
    return gr.Group(visible=True)


def make_group_invisible():
    return gr.Group(visible=False)
