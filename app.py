
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.metrics import accuracy_score
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import pandas as pd
import cv2
import numpy as np
import plotly.express as px

EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

st.title("Facial Emotion Detection")
st.header("Demo")
model = tf.keras.models.load_model("emotion_detection_model_for_streamlit.h5")
f = st.file_uploader("Upload Image")
if f is not None: 
  file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(image, channels="BGR")
  resized = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)
  gray_1d = np.mean(resized, axis=-1)
  gray = np.zeros_like(resized)
  gray[:,:,0] = gray_1d
  gray[:,:,1] = gray_1d
  gray[:,:,2] = gray_1d
  normalized = gray/255
  model_input = np.expand_dims(normalized,0)
  scores = model.predict(model_input).flatten() 
  df = pd.DataFrame()
  df["Emotion"] = EMOTIONS
  df["Scores"] = scores
  px.bar(df, x='Emotion', y='Scores', title="Model scores for each emotion")
  prediction = model.predict(model_input) 
  print (EMOTIONS[prediction.argmax(axis=1)[0]])
  print (str(prediction.max()*100) + '% Confidence' )
