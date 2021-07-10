from io import BytesIO
import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import altair as alt
import requests
import os


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/FoodVision-EfficientNetB0.h5')
    return model

with st.spinner('Loading Model into Memory......'):
    model = load_model()


## Side bar
with st.sidebar:
    st.title('What is Food Vision?')
    st.write("""
    Food Vision is an end-to-end **CNN Image Classification Model** which identifies
    over 100 food classes.
    
    It is based on pre-trained Image Classification Model `EfficienNetB0` which is 
    trained on dataset `Food-101`.
    """)
    st.write("")
    st.write("")
    st.markdown(body='#### Choose Input type:')
    st.write("")
    method = st.sidebar.radio('', options=['Image', 'URL'])
    st.write("")
    st.write("")
    st.write("To know more about this app, visit [**Github**](https://www.github.com/mSounak/Food-Vision)")

# Get the labels of the Food101
labels=[]
with open('meta/labels.txt', 'rb') as f:
    reader = f.read().splitlines()
    
    for item in reader:
        labels.append(item.decode('utf-8'))

# Prepare the data for prediction
def load_prep_img(image, image_size=(224, 224), scale=False):
    
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, size=image_size)

    if scale:
        return tf.expand_dims(img/255., axis=0)
    else:
        return tf.expand_dims(img, axis=0)

# Predict the image
def predicting(image, model):
    image = load_prep_img(image)
    preds = model.predict(image)
    pred_class = labels[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    classes = []
    for x in range(5):
        classes.append(labels[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": classes,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df



# Main Body
st.title("Food Vision 🍜📸")
st.header("Check your food photos!")
if (method == 'Image'):
    # For Image option
    file = st.file_uploader('Please upload a food image', type=['png', 'jpeg', 'jpg'])

    if file:
        image = file.read()
        st.image(file, caption='Uploaded Image', use_column_width=True)

        st.write("")
        st.write("Predicted Class: ")
        with st.spinner('Classifying........'):
            pred_class, pred_conf, df = predicting(image, model)
            st.success(f'Prediction : {pred_class} | Confidence : {pred_conf*100:.2f}%')
            st.write(alt.Chart(df).mark_bar().encode(
                x='F1 Scores',
                y=alt.X('Top 5 Predictions', sort=None),
                color=alt.Color("color", scale=None),
                text='F1 Scores'
            ).properties(width=600, height=400))
    else:
        st.warning("Please upload an image")
        st.stop()

else:
    # For URL option
    url = st.text_input('Please enter image URL: ', value='https://kfoods.com/images1/newrecipeicon/club-sandwich_483.jpg')

    if url:
        content = requests.get(url).content

        image = Image.open(BytesIO(content))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("")
        st.write("Predicted Class: ")
        with st.spinner('Classifying........'):
            pred_class, pred_conf, df = predicting(content, model)
            st.success(f'Prediction : {pred_class} | Confidence : {pred_conf*100:.2f}%')
            st.write(alt.Chart(df).mark_bar().encode(
                x='F1 Scores',
                y=alt.X('Top 5 Predictions', sort=None),
                color=alt.Color("color", scale=None),
                text='F1 Scores'
            ).properties(width=600, height=400))
    else:
        st.warning("Please enter an URL")
        st.stop()
