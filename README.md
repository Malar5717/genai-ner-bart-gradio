<img width="1463" height="886" alt="image" src="https://github.com/user-attachments/assets/757ba4ea-626c-413d-b558-ad8637906a68" />## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

#### Name: Malar Mariam S
#### Register Number: 212223230118

### PROBLEM STATEMENT:
Unstructured textual data contains critical informational entities such as individuals, locations, and organizations. Manually identifying these entities from vast amounts of text is inefficient and error-prone. The goal is to build an automated pipeline using a fine-tuned transformer model for Token Classification (NER) and host it through a web interface to allow users to extract and visualize named entities from custom text inputs seamlessly.

### DESIGN STEPS:

#### STEP 1: Environment Setup and API Initialization
Import necessary operational modules (os, IPython, dotenv, gradio) and load the Hugging Face API key along with the model endpoint configuration from environment variables.

#### STEP 2: Inference Function Definition
Create an inference handler function ner() that receives raw string input, submits it to the backend get_completion endpoint hosting dslim/bert-base-NER, and formats the returned predictions into a structured dictionary containing text and entity tokens.

#### STEP 3: Gradio Interface & Deployment Configuration
Construct the interactive web GUI using gr.Interface, defining standard gr.Textbox inputs and gr.HighlightedText outputs along with preset examples. Launch the interface using specified port configurations and public link sharing (share=True).

### PROGRAM:
```py
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) 
hf_api_key = os.environ['HF_API_KEY']

API_URL = os.environ['HF_API_NER_BASE']
text = "Too many people buy things they don't need with money they don't have to impress people they don't know.  Read Rich Dad, Poor Dad."
get_completion(text, parameters=None, ENDPOINT_URL=API_URL)

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
    allow_flagging="never",
    examples=["My mothers name is Waheeda, and she is a great cook", "My name is Malar Mariam and I primarily work with Web Development and i love Bananas"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```


### OUTPUT:
<img width="1463" height="886" alt="image" src="https://github.com/user-attachments/assets/5c2b7f5e-5d6c-4bdf-a5b0-cda96ffcb1b1" />

### RESULT:
The prototype for Named Entity Recognition (NER) using the dslim/bert-base-NER fine-tuned transformer model was successfully developed and deployed via the Gradio web application framework.
