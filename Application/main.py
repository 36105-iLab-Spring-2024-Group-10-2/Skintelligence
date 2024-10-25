import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models

import gradio as gr

import openai

from transformers import ViltForQuestionAnswering, ViltProcessor

proj_dir = os.path.join(os.getcwd(), os.pardir)

# MLC model code
mlc_weights = os.path.join(proj_dir, 'Models', 'Multilabel Classification', 'multilabel_model_on_top10.pth')
mlc_columns_to_predict = ['Erythema', 'Plaque', 'Papule', 'Brown(Hyperpigmentation)', 'Scale', 'Crust', 'Yellow', 'White(Hypopigmentation)', 'Nodule', 'Erosion']

predicted_annotations = None

# MLC model definition
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesionModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
mlc_model = SkinLesionModel(num_classes=len(mlc_columns_to_predict)).to("cpu")
mlc_model.load_state_dict(torch.load(mlc_weights, map_location=torch.device("cpu"), weights_only=True))
mlc_model.eval()

# Function to predict medical annotations using multilabel classifier
def mlc_predict(image):
    global predicted_annotations, mlc_model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_image = transform(image)
    outputs = mlc_model(transformed_image.unsqueeze(0))
    predicted_annotations = (torch.sigmoid(outputs) > 0.4).int()

# VQA model code
vqa_binary_column_names = ['Papule', 'Plaque', 'Crust', 'White(Hypopigmentation)', 'Erosion', 'Nodule', 'Scale', 'Brown(Hyperpigmentation)', 'Erythema', 'Yellow']
index_mapping = [mlc_columns_to_predict.index(column_name) for column_name in vqa_binary_column_names]
df = pd.read_csv(os.path.join(proj_dir, 'Data', 'Final', 'Final Single Answer Dataset.csv'))
unique_answers = df['answer'].unique()
answer_to_idx = {answer: idx for idx, answer in enumerate(unique_answers)}

diagnosis_question = 'Which skin condition is observed in this image?'

# Defining Model Class with Dropout Layer
class ViltForQuestionAnsweringWithBinary(nn.Module):
    # Add num_binary_features
    def __init__(self, model_name, num_labels, num_binary_features):
        super(ViltForQuestionAnsweringWithBinary, self).__init__()
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.1)
        # Updated classifier to incorporate binary features
        self.classifier = nn.Linear(self.model.config.hidden_size + num_binary_features, num_labels)

    def forward(self, input_ids, pixel_values, attention_mask, binary_variables):
        outputs = self.model.vilt(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]

        # Concatenate pooled output with binary variables
        combined_output = torch.cat((pooled_output, binary_variables), dim = 1)

        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)
        return logits

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")    
vqa_model = ViltForQuestionAnsweringWithBinary("dandelin/vilt-b32-finetuned-vqa", num_labels=len(df['answer'].unique()), num_binary_features=len(vqa_binary_column_names))
vqa_weights = os.path.join(proj_dir, 'Models', 'VQA', 'vilt_skincap_model_with_binary_split_single_answer_10_Annot_Epoch_20.pth')
vqa_model.load_state_dict(torch.load(vqa_weights, map_location=torch.device('cpu'), weights_only=True))

# Form the answer sentence using OpenAI
def form_answer(question, predicted_answer, uncertainty=0):
    openai.api_key = "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"
    
    if uncertainty > 1.5:
        prompt = f"For the provided image and the following question: {question}, Skintelligence model does not have the right response. Either the image may not be related to skin diseases or the model might have been trained on this disease or question. Form 1 to 2 line response for the model."
    elif question == diagnosis_question:
        prompt = f"For the provided image and the following question: {question}, the predicted answers are: {predicted_answer}. Mention all the possible diagnoses and form 3 to 4 line response for the model."
    else:
        prompt = f"For the provided image and the following question: {question}, the predicted answer is: {predicted_answer}. Form a one line response. If the predicted answer does not make sense for the question, provide your own one line answer with the text 'Generated by ChatGPT:' prepended to it."

    print(f'Predicted answer: {predicted_answer}')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an Answer Expander. Your task is to take the question and brief answers (1-3 words) and convert them into clear, complete responses that provide more context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        print(f"GPT response: {response['choices'][0]['message']['content'].strip().replace('Generated by ChatGPT:', '<i>Generated by ChatGPT:</i>')}")
        return response['choices'][0]['message']['content'].strip().replace('Generated by ChatGPT:', '<i>Generated by ChatGPT:</i>')
    except Exception as e:
        return f"Error: {str(e)}"
    
def predict_with_uncertainty(image, question, binary_variables, uncertainty_samples=10):
    global vqa_model, processor, unique_answers

    encoding = processor(image, question, return_tensors="pt", padding=True)

    vqa_model.train()    

    logits_list = []
    for _ in range(uncertainty_samples):
        outputs = vqa_model(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], attention_mask=encoding['attention_mask'], binary_variables=binary_variables)
        logits_list.append(outputs)

    vqa_model.eval()
    logits = torch.stack(logits_list)
    mean_logits = logits.mean(dim=0)
    std_logits = logits.std(dim=0)

    uncertainty = std_logits.mean().item()

    topk_values, topk_indices = torch.topk(mean_logits, 3, dim=-1)
    #topk_values = topk_values.squeeze(0)
    topk_indices = topk_indices.squeeze(0)

    #top3_answers = [(unique_answers[idx], value.item()) for idx, value in zip(topk_indices, topk_values)]
    top3_answers = [unique_answers[idx] for idx in topk_indices]
    response = form_answer(question, top3_answers, uncertainty)

    return response

def predict(image, question, binary_variables):
    global vqa_model, processor, unique_answers

    encoding = processor(image, question, return_tensors="pt", padding=True)

    vqa_model.eval()
    logits = vqa_model(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], attention_mask=encoding['attention_mask'], binary_variables=binary_variables)

    predicted_answer = unique_answers[logits.argmax(-1).item()]
    response = form_answer(question, predicted_answer)

    return response
    
# Function to predict vqa response using ViLT model
def vqa_response(image, question=None, chathistory=None):
    global predicted_annotations, answer_to_idx
    binary_variables = predicted_annotations

    image = image.convert("RGB")
    image = image.resize((384, 384))

    if not question:
        question = diagnosis_question
        response = predict_with_uncertainty(image=image, question=question, binary_variables=binary_variables, uncertainty_samples=10)
    else:
        response = predict(image=image, question=question, binary_variables=binary_variables)
        chathistory.append([question, response])
        return chathistory, ''

    return response

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

#gradio-image-box {
    border: 2px solid #00acc1;
    border-radius: 12px;
    transition: border-color 0.3s ease;
    max-height: 300px;
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
            <h1 style="font-size: 2.5em; color: #00acc1; font-family: 'Poppins', sans-serif;">
                Welcome to Skintelligence
            </h1>
        </div>
        """)

    with gr.Row() as interface_row: #visible=False
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Skin Image", elem_id="gradio-image-box", sources=['upload', 'webcam', 'clipboard'])
            with gr.Row():
                with gr.Column(min_width=150):
                    diagnosis_button = gr.Button("Get Diagnosis", elem_id="gradio-button")
                with gr.Column(min_width=150):
                    clear_button = gr.Button("Clear", elem_id="gradio-clear-button")
            with gr.Row():
                with gr.Column():
                    output_predicted = gr.Textbox(label="Skintelligence Predicted Diagnosis", elem_id="gradio-outputs")
            
        with gr.Column():
            #chatbot_history = gr.State([])
            chat_input = gr.Textbox(lines=1, placeholder="Ask a question related to the image", label="Chat with Assistant", elem_id="chat-input", interactive=True)
            chat_submit_button = gr.Button("Send", elem_id="chat-button")
            chat_output = gr.Chatbot(label="Assistant Response", elem_id="chat-output", height=300)  # Increased height for better visibility
            
        image_input.upload(mlc_predict, inputs=image_input, outputs=predicted_annotations)

        diagnosis_button.click(vqa_response, inputs=image_input, outputs=output_predicted)
        clear_button.click(lambda: (None, "", "", ""), inputs=[], outputs=[image_input, output_predicted, chat_input, chat_output])

        chat_submit_button.click(vqa_response, inputs=[image_input, chat_input, chat_output], outputs=[chat_output, chat_input])
        chat_input.submit(vqa_response, inputs=[image_input, chat_input, chat_output], outputs=[chat_output, chat_input])

demo.launch(share=False)