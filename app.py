import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image 

img_file = st.file_uploader("Choose a image file",type=["png","jpg","jpeg"])

if img_file is not None:
    progress_text="Generating Classification of the image."
    with st.spinner(progress_text):      
         model=load_model('model_vgg19.h5')
         img=image.load_img(img_file,target_size=(224,224))
         st.write('Uploaded Image:')
         st.write(img_file)
         x=image.img_to_array(img)
         x=x/255
         x=np.expand_dims(x,axis=0)
         img_data=preprocess_input(x)
         print(img_data.shape)
         pr=model.predict(img_data)
         a=np.argmax(pr,axis=1)
         if(a==1):
            st.write('Unifected')
         else:
             st.write('Infected')