import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("greencare.h5")

# Define class names
Classes = [
    "Apple_scab", "healthy_Apple", "Apple_Blotch_Fungus", "Northern_corn_leaf_blight",
    "healthy_Corn", "early_blight_potato", "Late_blight_Potato", "healthy_Potato",
    "Apple_Rot", "Apple_Scab", "Tomato_Bacterial_spot", "early_blight_Tomato", "Late_blight_Tomato",
    "Tomato_mosaic_virus", "healthy_Tomato", "Invalid"
]

# Define a function to preprocess the image
def prepare(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255
    return np.expand_dims(x, axis=0)

# TensorFlow Model Prediction
def model_prediction(test_image):
    img = Image.open(test_image)
    img = img.resize((256, 256))  # Resize to match the model's input size
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    input_arr = input_arr / 255.0  # Normalize
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Crop Monitor")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Crop Monitoring System! 🌿🔍
    
    Our mission is to help in identifying crop diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Crop Monitoring System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            result_index = model_prediction(test_image)
            predicted_class = Classes[result_index]
            st.success(f"Model is predicting it's {predicted_class}")
