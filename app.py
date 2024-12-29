import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_utils import load_model, predict  # assuming you've defined the loading and prediction logic

# Set up the page config
st.set_page_config(page_title="Fake Currency Classification", page_icon="💵")

# Title and description
st.title("Fake Currency Classification")
st.subheader("Upload an image of the Rs. 100 note (front side) to check if it is Real or Fake!")

# Image Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Ensure image is converted to RGB (remove alpha channel if present)
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Preprocess the image and perform prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model
    model = load_model("pretrained_vit_model.pth", device="cuda" if torch.cuda.is_available() else "cpu").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    class_names = ['100_Fake_Front', '100_Real_Front']

    # Perform prediction
    class_label, probability = predict(
        model, image_tensor, class_names,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Display prediction result with large and bold text
    st.markdown(f"### **Prediction: {class_label}**")
    st.markdown(f"#### **Probability: {probability * 100:.2f}%**")

# Upcoming Features Section
st.markdown("""
    <h2 style="text-align: center; color: white;">Upcoming Features</h2>
    <p style="text-align: center; color: white;">We are constantly working to improve this application. Here are some exciting new features that will be added soon:</p>
""", unsafe_allow_html=True)

# Feature 1: Other currency notes classification
st.subheader("1. Classifying Other Currency Notes")
st.image("worldclass.png", caption="All other currencies", use_container_width=True)

# Feature 2: Multi-class Classification
st.subheader("2. Multi-class Classification")
st.image("Indiancurrencynotes.png", caption="At a time many notes classification", use_container_width=True)

# Feature 3: Real-time Detection
st.subheader("3. Real-time Detection")
st.image("realtime.png", caption="Real-time Detection", use_container_width=True)

# Contact and Feedback Section
st.markdown("""
    <h2 style="text-align: center; color: white;">Contact & Feedback</h2>
    <p style="text-align: center; color: white;">Have suggestions or feedback? Feel free to reach out to us!</p>
""", unsafe_allow_html=True)
st.text_input("Your Name")
st.text_area("Your Feedback")
# Submit button for feedback
if st.button("Submit Feedback"):
    st.success("Successfully submitted your feedback! Thank you for your input.")

# Footer
st.markdown("""
    <footer style="text-align: center; color: white; padding: 20px;">
        &copy; 2024 Fake Currency Classification App
    </footer>
""", unsafe_allow_html=True)
