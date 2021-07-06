import streamlit as st
import tensorflow as tf
from PIL import Image
import imageio
import os


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Food Image Classifier")


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/FoodVision-EfficientNetB0.h5')
    return model

with st.spinner('Loading Model into Memory......'):
    model = load_model()

labels=[]
with open('meta/labels.txt', 'rb') as f:
    reader = f.read().splitlines()
    
    for item in reader:
        labels.append(item.decode('utf-8'))


def load_prep_img(image, image_size=(224, 224), scale=False):
    
    img = imageio.imread(image)
    img = tf.image.resize(img, size=image_size)

    if scale:
        return tf.expand_dims(img/255., axis=0)
    else:
        return tf.expand_dims(img, axis=0)

file = st.file_uploader('Please upload a food image', type=['png', 'jpeg', 'jpg'])

if file is not None:
    
    image = Image.open(file)
    st.image(file, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Predicted Class: ")
    with st.spinner('Classifying........'):
        img = load_prep_img(file)
        probs = model.predict(img)
        pred = labels[probs.argmax()]
        st.write(pred)
    
        



    # st.write("""
    #         # Food Classifier
    #         """)

    # uploaded_file = st.file_uploader('Please upload a food image', type=['png', 'jpeg', 'jpg'])

    # if uploaded_file:
    #     st.image(uploaded_file, use_column_width=True)
    #     read_img = ip.load_prep_img(uploaded_file)

    #     pred,_ = ip.prediction(read_img)

    #     string = "This image is most likely: " + pred
    #     st.success(string)
    # else:
    #     st.text("Please upload an image file")

