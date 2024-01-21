import gradio as gr

def change_tab():
        return gr.Tabs(selected=1)

def show_json(item):
      pass
    
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