import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models


import gradio as gr

#import openai

#from sklearn.utils.class_weight import compute_class_weight
#from transformers import ViltForQuestionAnswering, ViltProcessor

proj_dir = os.path.join(os.getcwd(), os.pardir)
mlc_weights = os.path.join(proj_dir, 'Models', 'Multilabel Classification', 'multilabel_model_on_top10.pth')
columns_to_predict = ['Erythema', 'Plaque', 'Papule', 'Brown(Hyperpigmentation)', 'Scale', 'Crust', 'Yellow', 'White(Hypopigmentation)', 'Nodule', 'Erosion']

# Model definition:
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesionModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
#print(mlc_model)
mlc_model = SkinLesionModel(num_classes=len(columns_to_predict)).to("cpu")
mlc_model.load_state_dict(torch.load(mlc_weights, map_location=torch.device("cpu"), weights_only=True))
mlc_model.eval()

def mlc_predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_image = transform(image)
    outputs = mlc_model(transformed_image.unsqueeze(0))
    return torch.sigmoid(outputs), torch.sigmoid(outputs) > 0.5

    #return f'Predicted answer: {outputs}'


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

    #with gr.Column():
    #    gr.Markdown("""
    #    <div style="text-align: center;">
    #        <h1 style="font-size: 5em; color: #00acc1; font-family: 'Poppins', sans-serif;">
    #            Welcome to Skintelligence
    #        </h1>
    #        <p style="font-size: 1.8em; color: #cfd8dc; font-family: 'Roboto', sans-serif; max-width: 900px; margin: 0 auto;">
    #            The future of dermatology is here! Upload an image of any skin condition, ask your question, and let our cutting-edge AI analyze and provide a smart, intuitive diagnosis with a vivid explanation. Revolutionizing skin health, one scan at a time.
    #        </p>
    #    </div>
    #    """)
    #    start_button = gr.Button("Start Your Diagnosis")

    with gr.Row() as interface_row: #visible=False
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Skin Image", elem_id="gradio-image-box", sources=['upload', 'webcam', 'clipboard'])
            question_input = gr.Textbox(lines=2, placeholder="Ask a question about the skin condition", label="Your Question", elem_id="gradio-textbox")
            submit_button = gr.Button("Get Diagnosis", elem_id="gradio-button")
            clear_button = gr.Button("Clear", elem_id="gradio-clear-button")

        with gr.Column():
            output_predicted = gr.Textbox(label="Skintelligence Predicted Answer", elem_id="gradio-outputs")
            output_vivid = gr.Textbox(label="Vivid Description", elem_id="gradio-outputs")

        image_input.upload(mlc_predict, inputs=image_input, outputs=[output_predicted, output_vivid])

        #submit_button.click(predict, inputs=[image_input, question_input], outputs=[output_predicted, output_vivid])

        #clear_button.click(lambda: (None, "", "", ""), inputs=[], outputs=[image_input, question_input, output_predicted, output_vivid])

    #start_button.click(lambda: gr.update(visible=False), outputs=[start_button])
    #start_button.click(lambda: gr.update(visible=True), outputs=[interface_row])

demo.launch(share=False)