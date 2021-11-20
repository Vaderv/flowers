#%%
import streamlit as st
import pandas as pd
import numpy as np
import pickle
#%%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']    
df = pd.read_csv(filepath_or_buffer=url, header=None, sep=',', names=names)   
#%%
st.title('Iris Flower Classifier')
st.markdown("""Predict the species of an Iris flower using sepal and petal measurements.""")

st.sidebar.header('Input Features')

sepal_length = st.sidebar.slider(
                label='Sepal Length',
                min_value=float(df['sepal-length'].min()),
                max_value=float(df['sepal-length'].max()),
                value=float(round(df['sepal-length'].mean(), 1)),
                step=0.1)

sepal_width = st.sidebar.slider(
                label='Sepal Width',
                min_value=float(df['sepal-width'].min()),
                max_value=float(df['sepal-width'].max()),
                value=float(round(df['sepal-width'].mean(), 1)),
                step=0.1)

petal_length = st.sidebar.slider(
                label='Petal Length',
                min_value=float(df['petal-length'].min()),
                max_value=float(df['petal-length'].max()),
                value=float(round(df['petal-length'].mean(), 1)),
                step=0.1)

petal_width = st.sidebar.slider(
                label='Petal Width',
                min_value=float(df['petal-width'].min()),
                max_value=float(df['petal-width'].max()),
                value=float(round(df['petal-width'].mean(), 1)),
                step=0.1)

loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))

X = [[sepal_length, sepal_width, petal_length, petal_width]]
y_pred = loaded_model.predict(X)
df_pred = pd.DataFrame({'Species': ['Virginica', 'Versicolor', 'Setosa']})
y_pred_df = pd.DataFrame(y_pred).rename(columns={0:'Flower Type'})
st.write(y_pred_df)
#%%






























