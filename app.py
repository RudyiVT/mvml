import joblib
import pandas as pd
import gradio as gr

cl = joblib.load("models/model.pkl")


def estimate(title=None, author=None, text=None):
    df = pd.DataFrame([[title, author, text]], columns=['title', 'author', 'text'])
    return cl.predict_proba(df)[0, 1]


with gr.Blocks() as demo:
    title = gr.Textbox(label="title")
    author = gr.Textbox(label="author")
    text = gr.Textbox(label="text")

    output = gr.Textbox(label="Probability")
    greet_btn = gr.Button("Score")
    greet_btn.click(fn=estimate, inputs=[title, author, text], outputs=output)

demo.launch(share=True)
