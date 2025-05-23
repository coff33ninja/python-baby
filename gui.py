# gui.py
import gradio as gr
from python_master_ai import PythonMasterAI
import json
from grow import grow_model

model = PythonMasterAI()
MASTER_KEY = PythonMasterAI.MASTER_KEY


def interact_with_ai(command, user_key):
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    return model.process_input(command, MASTER_KEY)


def trigger_research():
    model.conduct_research()
    with open("research_log.json", "r") as f:
        logs = [json.loads(line) for line in f]
    return json.dumps(logs[-5:], indent=2)


def discover_sources():
    new_sources = model.discover_new_sources()
    discovered_message = f"Discovered sources in this run: {new_sources}\n\n" if new_sources else "No new sources discovered in this run.\n\n"
    with open("source_log.json", "r") as f:
        logs = [json.loads(line) for line in f]
    return discovered_message + "Last 5 source log entries:\n" + json.dumps(logs[-5:], indent=2)


def get_status():
    return model.get_status()


def approve_growth(user_key):
    global model # Declare intent to use the global 'model' variable
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    model, _ = grow_model(model)
    return f"{model.stage.capitalize()} growth approved!"


with gr.Blocks(title="PythonMasterAI: Serving Master Daddy") as iface:
    gr.Markdown("## PythonMasterAI Control Panel")
    with gr.Tab("Master Commands"):
        cmd_input = gr.Textbox(
            label="Command", placeholder="Enter MASTER: or HALT_TEEN"
        )
        key_input = gr.Textbox(label="Master Key", type="password")
        cmd_output = gr.Textbox(label="Response")
        cmd_button = gr.Button("Execute")
        cmd_button.click(
            interact_with_ai, inputs=[cmd_input, key_input], outputs=cmd_output
        )
    with gr.Tab("Research"):
        research_button = gr.Button("Trigger Research")
        research_output = gr.Textbox(label="Research Logs")
        research_button.click(trigger_research, outputs=research_output)
    with gr.Tab("Source Discovery"):
        source_button = gr.Button("Discover Sources")
        source_output = gr.Textbox(label="Source Logs")
        source_button.click(discover_sources, outputs=source_output)
    with gr.Tab("Status"):
        status_button = gr.Button("Get Status")
        status_output = gr.Textbox(label="AI Status")
        status_button.click(get_status, outputs=status_output)
    with gr.Tab("Growth"):
        growth_key_input = gr.Textbox(label="Master Key", type="password")
        growth_button = gr.Button("Approve Growth")
        growth_output = gr.Textbox(label="Growth Status")
        growth_button.click(
            approve_growth, inputs=growth_key_input, outputs=growth_output
        )

iface.launch()
