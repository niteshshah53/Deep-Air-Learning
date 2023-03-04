# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 00:24:35 2022

@author: rider
"""
import numpy as np
import pickle
import streamlit as st
import warnings
import matplotlib.pyplot as plt

# Loading the saved regression model
loaded_model = pickle.load(open('Algorithm.pkl','rb'))

# Setting the background color and page layout
st.set_page_config(page_title='Deep Air Learning', page_icon=":cloud:", layout="wide")

# Adding header image
from PIL import Image
header_image = Image.open('a.jpg')
st.image(header_image, width=1600)

def main():
    # Giving a title
    st.title('Air Quality Index Prediction')
    st.write("Enter the following air pollutant concentrations to get the Air Quality Index")

    # Getting input data from the user
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        so2 = st.slider('SO2', min_value=0, max_value=100, value=50)
    with col2:
        no2 = st.slider('NO2', min_value=0, max_value=100, value=50)
    with col3:
        pm10 = st.slider('PM10', min_value=0, max_value=100, value=50)
    with col4:
        pm2_5 = st.slider('PM2.5', min_value=0, max_value=100, value=50)

    # Centering the 'Calculate AQI' button
    button_container = st.container()
    with button_container:
        st.write('')
        st.write('')
        st.write('')
        col5, col6, col7 = st.columns([1, 1, 3])
        with col7:
            if st.button('Calculate AQI', key='calculate',):
                # Predicting the AQI
                input_data = [so2, no2, pm10, pm2_5]
                x = loaded_model.predict([input_data])[0]

                # Displaying the AQI and air quality status
                st.write("Air Quality Index (AQI) is: ", x)

                if x <= 50:
                    st.write("Air Quality is Excellent and ideal for normal outdoor activities :sunglasses:")
                    image = Image.open('b.jpg')
                elif x <= 100:
                    st.write("Air Quality is Moderate :neutral_face:")
                    image = Image.open('c.jpg')
                elif x <= 200:
                    st.write("Air Quality is Poor :mask:")
                    image = Image.open('d.jpg')
                elif x <= 300:
                    st.write("Air Quality is Unhealthy :face_with_thermometer:")
                    image = Image.open('e.jpg')
                elif x <= 400:
                    st.write("Air Quality is Very Unhealthy :dizzy_face:")
                    image = Image.open('f.jpg')
                else:
                    st.write("Air Quality is Hazardous :warning:")
                    image = Image.open('g.jpg')

                # Displaying the air quality image
                st.image(image, width=400)

                # Displaying the histogram of AQI values
                st.write("Histogram of AQI values:")
                aqi_values = np.random.normal(x, 20, 100)
                fig, ax = plt.subplots()
                ax.hist(aqi_values, bins=20, alpha=0.5)
                ax.set_xlabel('AQI')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

if __name__ == '__main__':
    st.markdown("""<style> body { background-color: #F8F8FF;} </style>""", unsafe_allow_html=True)
    main()
