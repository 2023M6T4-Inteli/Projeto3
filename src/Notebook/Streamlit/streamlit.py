import streamlit as st
import pandas as pd
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

from sklearn.feature_extraction.text import CountVectorizer
import ast
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import pad_sequences
import tensorflow as tf
from keras.metrics import Recall
from sklearn.metrics import recall_score


import nltk
import spacy
import gensim
import pickle
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

# Carrega o modelo treinado
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# def vectorize_text(text):
#     #TFIDF

# Classifica as linhas do CSV
def classify_rows(model, df):
    df2 = df
    df2.insert(loc=1, column='classificacao', value=model.predict(df2.iloc[:,1:50]))
    return df2

def sentiment_chart(df2):
    # Contar os tipos de sentimento
    count_sentiment = df2['classificacao'].value_counts()
    # Criar o gráfico de pizza
    plt.figure(figsize=(2, 2), facecolor='#11111E')
    count_sentiment.plot(kind='pie', autopct='%1.1f%%', textprops={'color': 'white'})
    plt.ylabel('')
    st.pyplot(plt)


# Interface do Streamlit
def main():
    # Título e descrição
    st.set_page_config(page_title="BT-G3", page_icon=":bar_chart:", layout="wide")
    st.title("BT-G3")
    st.write("Carregue um arquivo CSV para ser classificado.")

    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Carregue o arquivo CSV", type="csv")

    st.write("---")


    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.subheader(".CSV carregado:")
        st.dataframe(df.head(5), use_container_width=True)

        st.write("---")

        # Carrega o modelo
        model_path = "naivebayes_word2vec_cbow_sprint3.pkl"  # Caminho para o arquivo .pkl do modelo treinado
        model = load_model(model_path)
        

        st.subheader("Resultado da classificação:")

        # Classifica as linhas
        df2 = classify_rows(model, df)

        # separa a interface em duas colunas
        columns = st.columns((2,1))

        with columns[0]:
            st.subheader(".CSV classificado:")
            st.dataframe(df2.iloc[:6,0:2], use_container_width=True)    

        with columns[1]:
            st.subheader("Palavras mais frequentes:")
            st.write(df2.columns)
            
        with st.container():
            sentiment_chart(df2)

# Executa o aplicativo
if __name__ == '__main__':
    main()
