import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="C-Frame Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .uploadedImage {
        max-width: 500px;
        margin: 20px auto;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load the trained model"""
    return load_model('C-frame80.h5')

def preprocess_image(img):
    """Preprocess the uploaded image for model prediction"""
    # Convert image to RGB if it's not already
    img = img.convert('RGB')
    # Resize image to 224x224
    img = img.resize((224, 224))
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def main():
    # Header
    st.title("C-Frame Image Classifier 🖼️")
    st.write("Upload an image to classify it as Categorized or Uncategorized")

    # Load model
    try:
        model = load_classifier()
        st.success("Model Siap Digunakan!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read and display image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Add a predict button
            if st.button("Predict"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_img = preprocess_image(img)
                    
                    # Make prediction
                    prediction = model.predict(processed_img)
                    class_names = ['Star Seller', 'Underrated', 'Random Pictures']
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    # Display results
                    st.markdown(f"### Prediction: {predicted_class}")
                    st.markdown(f"### Confidence: {confidence:.2f}%")

                    # Display confidence bar
                    st.progress(float(confidence) / 100)

                    # Additional information based on prediction
                    if predicted_class == 'Star Seller':
                        st.success("This item is categorized as a Star Seller ⭐")
                    elif predicted_class == 'Underrated':
                        st.info("This item is considered Underrated 🔍")
                    else:
                        st.warning("This is a Random Picture 📷 – may not belong to a specific category.")
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please make sure you upload a valid image file.")

    # Add information about the model
    with st.expander("About this app"):
        st.write("""
        This app uses a MobileNetV2-based deep learning model to classify images into two categories:
        - **Categorized**: Images that are properly categorized
        - **Uncategorized**: Images that need categorization
        
        The model was trained on a custom dataset and achieves good accuracy in distinguishing between these categories.
        """)

if __name__ == "__main__":
    main()
