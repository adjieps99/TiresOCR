import streamlit as st
from PIL import Image
import func_ as f

import os

st.write("""
         # Text Detection on Tires Demo
         """
         )

st.write("This is an image detection web app to predict the object in the image")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    filename = "uploaded.jpg"
    image.save(filename)
    result, img_result = f.predict(filename)
    os.remove(filename)
    st.image(img_result, width=480,channels="RGB")
    st.write(result)