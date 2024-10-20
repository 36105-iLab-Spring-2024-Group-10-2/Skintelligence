import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import openai
import gradio as gr
from sklearn.utils.class_weight import compute_class_weight
from transformers import ViltForQuestionAnswering, ViltProcessor

df = pd.read_csv('../Data/Final/Final Complete Dataset.csv')
unique_answers = df['answer'].unique()
answer_to_idx = {answer: idx for idx, answer in enumerate(unique_answers)}
class_weights = compute_class_weight(class_weight='balanced', classes=np.array(list(answer_to_idx.values())), y=df['answer'].map(answer_to_idx).values)
weights = torch.tensor(class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Defining Model Class with Dropout Layer
class ViltForQuestionAnsweringWithDropout(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ViltForQuestionAnsweringWithDropout, self).__init__()
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, pixel_values, attention_mask):
        outputs = self.model.vilt(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Function to Predict with Uncertainty
def predict_with_uncertainty(image, question, num_samples=100):
    image = image.convert("RGB")
    image = image.resize((384, 384))

    model.train()
    encoding = processor(image, question, return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logits_list = []
    for _ in range(num_samples):
        outputs = model(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], attention_mask=encoding['attention_mask'])
        logits_list.append(outputs)

    model.eval()
    logits = torch.stack(logits_list)
    mean_logits = logits.mean(dim=0)
    std_logits = logits.std(dim=0)

    predicted_answer_idx = mean_logits.argmax(-1).item()
    uncertainty = std_logits.mean().item()

    return unique_answers[predicted_answer_idx], uncertainty


openai.api_key = "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"

# Fetching Detailed Explanation from OpenAI
def get_detailed_answer(predicted_answer):
    prompt = f"Please provide a vivid and detailed explanation about the skin condition '{predicted_answer}'."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"
    
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnsweringWithDropout("dandelin/vilt-b32-finetuned-vqa", num_labels=len(df['answer'].unique()))
model.load_state_dict(torch.load('vilt_skincap_model.pth', map_location=torch.device('cpu')))
#model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Gradio Interface for Skintelligence

def predict(image, question):
    predicted_answer, uncertainty = predict_with_uncertainty(image, question)

    if uncertainty > 1.5:
        return f"Skintelligence Predicted Answer: {predicted_answer}, but the model is uncertain.", ""

    detailed_answer = get_detailed_answer(predicted_answer)

    return f"Skintelligence Predicted Answer: {predicted_answer}", f"Vivid Description: {detailed_answer}"

# CSS Styling for Gradio Interface
css = """
body {
    background: linear-gradient(to right, #1a1f36, #283c86);
    color: #ffffff;
    font-family: 'Roboto', sans-serif;
}

.gradio-container {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.gradio-title {
    font-family: 'Poppins', sans-serif;
    font-size: 4em;
    text-align: center;
    color: #00acc1;
    margin-top: 100px;
}

.gradio-description {
    font-family: 'Lato', sans-serif;
    font-size: 1.5em;
    text-align: center;
    margin-bottom: 50px;
    color: #cfd8dc;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

.gradio-inputs, .gradio-outputs {
    margin-top: 30px;
    border-top: 2px solid #00acc1;
    padding-top: 20px;
}

.gradio-button {
    background-color: #00acc1;
    color: #ffffff;
    font-size: 1.3em;
    padding: 12px 30px;
    border-radius: 8px;
    transition: box-shadow 0.3s ease;
    display: block;
    margin: 40px auto;
}

.gradio-button:hover {
    box-shadow: 0 0 20px rgba(0, 172, 193, 0.8);
}

.gradio-clear-button {
    background-color: #ff6f61;
    color: #ffffff;
    font-size: 1.1em;
    padding: 10px 25px;
    border-radius: 8px;
    transition: box-shadow 0.3s ease;
    display: block;
    margin: 20px auto;
}

.gradio-clear-button:hover {
    box-shadow: 0 0 20px rgba(255, 111, 97, 0.8);
}

.gradio-image-box {
    border: 2px solid #00acc1;
    border-radius: 12px;
    transition: border-color 0.3s ease;
}

.gradio-image-box:hover {
    border-color: #00acc1;
}

.gradio-textbox {
    font-size: 1.1em;
    padding: 15px;
    background-color: #283c86;
    border-radius: 10px;
    color: #ffffff;
    border: 1px solid #00acc1;
}

.gradio-outputs textarea {
    font-size: 1.2em;
    line-height: 1.6;
    background-color: #1a1f36;
    color: #ffffff;
    border: 1px solid #00acc1;
    padding: 20px;
    border-radius: 10px;
}
"""

# Launching Gradio Interface
with gr.Blocks(css=css) as demo:

    with gr.Column():
        gr.Markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 5em; color: #00acc1; font-family: 'Poppins', sans-serif;">
                Welcome to Skintelligence
            </h1>
            <p style="font-size: 1.8em; color: #cfd8dc; font-family: 'Roboto', sans-serif; max-width: 900px; margin: 0 auto;">
                The future of dermatology is here! Upload an image of any skin condition, ask your question, and let our cutting-edge AI analyze and provide a smart, intuitive diagnosis with a vivid explanation. Revolutionizing skin health, one scan at a time.
            </p>
        </div>
        """)
        start_button = gr.Button("Start Your Diagnosis")

    with gr.Row(visible=False) as interface_row:
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Skin Image", elem_id="gradio-image-box")
            question_input = gr.Textbox(lines=2, placeholder="Ask a question about the skin condition", label="Your Question", elem_id="gradio-textbox")
            submit_button = gr.Button("Get Diagnosis", elem_id="gradio-button")
            clear_button = gr.Button("Clear", elem_id="gradio-clear-button")

        with gr.Column():
            output_predicted = gr.Textbox(label="Skintelligence Predicted Answer", elem_id="gradio-outputs")
            output_vivid = gr.Textbox(label="Vivid Description", elem_id="gradio-outputs")

        submit_button.click(predict, inputs=[image_input, question_input], outputs=[output_predicted, output_vivid])

        clear_button.click(lambda: (None, "", "", ""), inputs=[], outputs=[image_input, question_input, output_predicted, output_vivid])

    start_button.click(lambda: gr.update(visible=False), outputs=[start_button])
    start_button.click(lambda: gr.update(visible=True), outputs=[interface_row])

demo.launch(share=True)