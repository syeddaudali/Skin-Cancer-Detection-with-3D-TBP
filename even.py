import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your Google Gemini API key from .env
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    st.error("Google API key not found in environment variables!")
else:
    genai.configure(api_key=google_api_key)

# Title of the app
st.title('Melanoma vs Benign Skin Lesion Detection')

# Define the model loading function with caching
@st.cache_resource
def load_model():
    # Load a pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: Melanoma vs Benign

    # Load the model's weights from file (use CPU)
    model.load_state_dict(torch.load('E:/Knowledge Streams/Mega Project/VS/model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    return model

# Load the trained model
model = load_model()

# Define image preprocessing transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Resizing the image to the same size as the model expects
        transforms.ToTensor(),           # Converting the image to a tensor
        transforms.Normalize(            # Normalizing the image tensor
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # Add a batch dimension

# Define function to get prediction from the model
def get_prediction(image_tensor):
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Define function to get Gemini response based on the prediction
def get_gemini_response(prediction):
    # Create a prompt based on the model's prediction
    if prediction == 0:
        prompt = "The model predicted the skin lesion as Benign. Please explain what this means and provide any relevant advice."
    else:
        prompt = "The model predicted the skin lesion as Malignant. Please explain what this means, provide advice, and suggest next steps."

    try:
        # Use the generate_text method with the prompt
        response = genai.generate_text(prompt=prompt)

        if response and 'generations' in response:
            return response['generations'][0]['text']  # Adjust based on the actual response format
        else:
            st.error("No response from Google Gemini.")
            return "No response"
    except Exception as e:
        st.error(f"Error generating response from Google Gemini: {str(e)}")
        return "Error"

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image in Streamlit
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_tensor = transform_image(image)

    # Get the prediction
    prediction = get_prediction(image_tensor)

    # Display the result
    if prediction == 0:
        st.write("Prediction: **Benign**")
    else:
        st.write("Prediction: **Malignant**")

    # Get and display the Gemini response
    gemini_response = get_gemini_response(prediction)
    st.write("Gemini Response:")
    st.write(gemini_response)
