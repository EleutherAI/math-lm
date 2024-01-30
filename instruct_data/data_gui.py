import yaml
import json
import os
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML, clear_output

styles = {
    'Accepted': 'background-color: rgba(0, 255, 0, 0.3);',  # Translucent green
    'Rejected': 'background-color: rgba(255, 0, 0, 0.3);',  # Translucent red
    'No judgement': 'background-color: rgba(0, 0, 0, 0.3);'
}

def display_item(data, judgement_file, index=0):
    if os.path.isfile(judgement_file):
        with open(judgement_file) as fle:
            judgement_map = yaml.safe_load(fle)
    else:
        judgement_map = dict()

    clear_output(wait=True)
    item = data[index]
    text_display = Markdown(item['text'])

    accept_button = widgets.Button(description="Accept")
    reject_button = widgets.Button(description="Reject")
    next_button = widgets.Button(description="Next")
    prev_button = widgets.Button(description="Previous")

    def display_verdict(verdict):
        verdict_display = HTML(f"<div style='{styles[verdict]}'>{verdict}</div>")
        display(verdict_display)

    def on_accept(b):
        judgement_map[item["id"]] = True
        with open(judgement_file, 'w') as fle:
            yaml.dump(judgement_map, fle)

        navigate(1)

    def on_reject(b):
        judgement_map[item["id"]] = False
        with open(judgement_file, 'w') as fle:
            yaml.dump(judgement_map, fle)
        navigate(1)

    def navigate(step):
        nonlocal index
        index = min(max(0, index + step), len(data) - 1)
        display_item(data, judgement_file, index)

    def display_buttons():
        button_box = widgets.HBox([accept_button, reject_button, prev_button, next_button])
        display(button_box)
    
    def display_location(index):
        display(Markdown(f"Index: {index}/{len(data)}"))

        display(Markdown(f"Post ID: {item['id']}"))
    
        num_judged = len(judgement_map.keys())
        num_accepted = sum(judgement_map.values())
        display(Markdown(f"Accepted: {num_accepted}, Judged: {num_judged}"))

    accept_button.on_click(on_accept)
    reject_button.on_click(on_reject)
    next_button.on_click(lambda b: navigate(1))
    prev_button.on_click(lambda b: navigate(-1))

    display_buttons()

    if item["id"] in judgement_map:
        verdict = 'Accepted' if judgement_map[item["id"]] else 'Rejected'
    else:
        verdict = 'No judgement'

    display(text_display)
    display_verdict(verdict)
    display_location(index)