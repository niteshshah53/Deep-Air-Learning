# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

#loading the saved regression model
loaded_model=pickle.load(open('C:/Users/rider/Desktop/MsAI/DeepAirLearning/Algorithm.pkl','rb'))

#loading the saved classification model
loaded_model2=pickle.load(open('C:/Users/rider/Desktop/MsAI/DeepAirLearning/Algorithm2.pkl','rb'))

input_data=[4.8,17.4,333,111]
print("AQI is : ",loaded_model.predict([input_data]))
print("AQI is classified as : ",loaded_model2.predict([input_data]))