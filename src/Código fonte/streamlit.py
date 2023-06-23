import streamlit as st
import pandas as pd
import pickle


import numpy as np
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from wordcloud import WordCloud
import re

# Carrega o modelo treinado
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Classifica as linhas do CSV
def classify_rows(model, df):
    df2 = df
    df2.insert(loc=1, column='classificacao', value=model.predict(df2.iloc[:,2:-1]))
    return df2


# Grafico de barras horizontal das classificações de sentimento
def sentiment_chart3(df2):
    df3 = df2
    replace_dict = {2: 'Positivo', 1: 'Neutro', 0: 'Negativo'}
    df3['classificacao'] = df3['classificacao'].replace(replace_dict)

    plt.figure(figsize=(18, 6), facecolor='#11111E') ##11111E
    plt.rcParams['text.color'] = '#FFFFFF'  # Definir a cor do texto como branco
    plt.rcParams['axes.facecolor'] = '#11111E'  # Definir a cor de fundo como #11111E

    count_sentiment = df2['classificacao'].value_counts()
    count_sentiment.plot(kind='barh', color=['#0000FF','#00FF00','#FF0000'])
    plt.xlabel('Quantidade', color='#FFFFFF')
    plt.ylabel('Sentimento', color='#FFFFFF')
    plt.title('Distribuição dos sentimentos', color='#FFFFFF')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')

    st.pyplot(plt)

def bow_dataframe(df2):
    tokenizer = Tokenizer() # usando o tokenizer da biblioteca do keras
    tokenizer.fit_on_texts(df2) # fitando o tokenizer com o que será passado como parâmetro
    wordCount = tokenizer.word_counts # pegando a contagem de palavras do tokenizer
    dfCountBoW = pd.DataFrame(list(wordCount.items())) # transformando em dataframe para melhor visualização
    dfCountBoW.rename(columns={0: "Palavra", 1:"Frequência"}, inplace=True) # renomeando as colunas
    final_df = dfCountBoW.sort_values(by=['Frequência'], ascending=False) # ordenando o dataframe
    return final_df

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
        model_path = "modelo_rf_novo.pkl"  # Caminho para o arquivo .pkl do modelo treinado
        model = load_model(model_path)
        

        st.subheader("Resultado da classificação:")

        # Classifica as linhas
        df2 = classify_rows(model, df)

        # separa a interface em duas colunas
        columns = st.columns((2,1))

        with columns[0]:
            st.subheader(".CSV classificado:")
            st.dataframe(df2.iloc[:11,0:2], use_container_width=True)    

        with columns[1]:
            st.subheader("Palavras mais frequentes:")
            st.dataframe(bow_dataframe(df2['texto_tratado']).head(10), use_container_width=True)
            
        with st.container():
            st.write('---')
            sentiment_chart3(df2)

# Executa o aplicativo
if __name__ == '__main__':
    main()
