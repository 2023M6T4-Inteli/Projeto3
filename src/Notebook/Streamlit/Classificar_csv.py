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
import emoji
from nltk.tokenize import TweetTokenizer


import nltk
import spacy
import gensim
import pickle
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

def tokenizer(comentarios):
  comentarios_tokenizados = []
  tk = TweetTokenizer()
  for comentario in comentarios:
    palavras = tk.tokenize(comentario.lower())
    comentarios_tokenizados.append(palavras)
  return comentarios_tokenizados

#### 4.2 Tratamento de emoji

def demojize_tokens(tokens):
  demojized_tokens = []
  for termo in tokens:
    demojized = [emoji.demojize(palavra) if emoji.emoji_count(palavra) > 0 else palavra for palavra in termo]
    demojized = [palavra.replace(":", "").replace("_", "") if any(c in palavra for c in [":", "_"]) else palavra for palavra in demojized]
    demojized = [palavra.replace("-", "_") if "-" in palavra else palavra for palavra in demojized]
    demojized_tokens.append(demojized)
  return demojized_tokens

#### 4.3 Remoção de Alfanuméricos


def removendo_alfanumericos(tokens):
  output_tokens = []
  for sentence in tokens:
      output_list = []
      for palavra in sentence:
          if palavra.strip(): # Verifica se a palavra não é uma string vazia
            palavra_sem_marcacao = re.sub((r'@\w*'), '', palavra)
            palavra_sem_hashtag = re.sub((r'#\w*'), '', palavra_sem_marcacao)
            palavra_sem_hyperlink = re.sub(r'https\S*', '', palavra_sem_hashtag)
            palavra_sem_www = re.sub(r'\bwww\.[^\s]*', '', palavra_sem_hyperlink)
            palavra_sem_numeros = re.sub((r'[0-9]'), '', palavra_sem_www)
            palavra_sem_btg = re.sub((r'\bbtg\b'), '', palavra_sem_numeros)
            palavra_sem_btgpactual = re.sub((r'\bpactual\b'), '', palavra_sem_btg)
            output_list.extend(re.findall(r'\w+', palavra_sem_btgpactual)) # analisar se não é melhor usar o append em vez de extend
      output_tokens.append(output_list)
  return output_tokens

#### 4.4 Tratamento de abreviações 


# Dicionário de gírias e abreviações para normalização
dicionario_girias = {
    'vc': 'você',
    'vcs':'você',
    'Vc': 'você',
    'pq': 'porque',
    'Pq': 'porque',
    'tbm': 'também',
    'q': 'que',
    'td': 'tudo',
    'blz': 'beleza',
    'flw': 'falou',
    'kd': 'cadê',
    'Gnt ': 'gente',
    'gnt ': 'gente',
    'to': 'estou',
    'mt': 'muito',
    'cmg': 'comigo',
    'ctz': 'certeza',
    'jah': 'já',
    'naum': 'não',
    'ta': 'está',
    'eh': 'é',
    'vdd': 'verdade',
    'vlw': 'valeu',
    'p': 'para',
    'sdds': 'saudades',
    'qnd': 'quando',
    'msm': 'mesmo',
    'fzr': 'fazer',
    's' : 'sim',
    'ss': 'sim',
    'Ss': 'sim',
    'pdc': 'pode crer',
    'n' : 'não',
    'nn': 'não',
    'Nn': 'não',
    'pls': 'please',
    'obg': 'obrigado',
    'agr': 'agora'
}

def comentarios_normalizados(tokens, dicionario_girias):
  tokens_normalizados = []

  for sentence in tokens:
    treated = []

    for palavra in sentence:
      if palavra in dicionario_girias:
          palavra_normalizada = dicionario_girias.get(palavra, palavra)
          treated.append(palavra_normalizada)
      else:
          treated.append(palavra)

    treated = [palavra.replace(' ', '') if '_' in palavra else palavra for palavra in treated]
    tokens_normalizados.append(treated)

  return tokens_normalizados

#### 4.5 Remoção de stopwords

##### Lista stopwords

stopwords = nltk.corpus.stopwords.words('portuguese')
len(stopwords)

new_stopwords = [ 'a', 'à', 'adeus', 'agora', 'aí', 'ainda', 'além', 'algo', 'alguém', 'algum', 'alguma', 'algumas', 'alguns', 'ali', 'ampla', 'amplas', 'amplo', 'amplos', 'ano', 'anos', 'ante', 'antes', 'ao', 'aos', 'apenas', 'apoio', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aqui', 'aquilo', 'área', 'as', 'às', 'assim', 'até', 'atrás', 'através', 'baixo', 'bastante', 'bem', 'boa', 'boas', 'bom', 'bons', 'breve', 'cá', 'cada', 'catorze', 'cedo', 'cento', 'certamente', 'certeza', 'cima', 'cinco', 'coisa', 'coisas', 'com', 'como', 'conselho', 'contra', 'contudo', 'custa', 'da', 'dá', 'dão', 'daquela', 'daquelas', 'daquele', 'daqueles', 'dar', 'das', 'de', 'debaixo', 'dela', 'delas', 'dele', 'deles', 'demais', 'dentro', 'depois', 'desde', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devendo', 'dever', 'deverá', 'deverão', 'deveria', 'deveriam', 'devia', 'deviam', 'dez', 'dezanove', 'dezasseis', 'dezassete', 'dezoito', 'dia', 'diante', 'disse', 'disso', 'disto', 'dito', 'diz', 'dizem', 'dizer', 'do', 'dois', 'dos', 'doze', 'duas', 'dúvida', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'embora', 'enquanto', 'entre', 'era', 'eram', 'éramos', 'és', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos', 'estão', 'estar', 'estas', 'estás', 'estava', 'estavam', 'estávamos', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estiveste', 'estivestes', 'estou', 'etc', 'eu', 'exemplo', 'faço', 'falta', 'favor', 'faz', 'fazeis', 'fazem', 'fazemos', 'fazendo', 'fazer', 'fazes', 'feita', 'feitas', 'feito', 'feitos', 'fez', 'fim', 'final', 'foi', 'fomos', 'for', 'fora', 'foram', 'fôramos', 'forem', 'forma', 'formos', 'fosse', 'fossem', 'fôssemos', 'foste', 'fostes', 'fui', 'geral', 'grande', 'grandes', 'grupo', 'há', 'haja', 'hajam', 'hajamos', 'hão', 'havemos', 'havia', 'hei', 'hoje', 'hora', 'horas', 'houve', 'houvemos', 'houver', 'houvera', 'houverá', 'houveram', 'houvéramos', 'houverão', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'la', 'lá', 'lado', 'lhe', 'lhes', 'lo', 'local', 'logo', 'longe', 'lugar', 'maior', 'maioria', 'mais', 'mal', 'mas', 'máximo', 'me', 'meio', 'menor', 'menos', 'mês', 'meses', 'mesma', 'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'mil', 'minha', 'minhas', 'momento', 'muita', 'muitas', 'muito', 'muitos', 'na', 'nada', 'não', 'naquela', 'naquelas', 'naquele', 'naqueles', 'nas', 'nem', 'nenhum', 'nenhuma', 'nessa', 'nessas', 'nesse', 'nesses', 'nesta', 'nestas', 'neste', 'nestes', 'ninguém', 'nível', 'no', 'noite', 'nome', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'nova', 'novas', 'nove', 'novo', 'novos', 'num', 'numa', 'número', 'nunca', 'o', 'obra', 'obrigada', 'obrigado', 'oitava', 'oitavo', 'oito', 'onde', 'ontem', 'onze', 'os', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'parece', 'parte', 'partir', 'paucas', 'pela', 'pelas', 'pelo', 'pelos', 'pequena', 'pequenas', 'pequeno', 'pequenos', 'per', 'perante', 'perto', 'pode', 'pude', 'pôde', 'podem', 'podendo', 'poder', 'poderia', 'poderiam', 'podia', 'podiam', 'põe', 'põem', 'pois', 'ponto', 'pontos', 'por', 'porém', 'porque', 'porquê', 'posição', 'possível', 'possivelmente', 'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'primeira', 'primeiras', 'primeiro', 'primeiros', 'própria', 'próprias', 'próprio', 'próprios', 'próxima', 'próximas', 'próximo', 'próximos', 'pude', 'puderam', 'quais', 'quáis', 'qual', 'quando', 'quanto', 'quantos', 'quarta', 'quarto', 'quatro', 'que', 'quê', 'quem', 'quer', 'quereis', 'querem', 'queremas', 'queres', 'quero', 'questão', 'quinta', 'quinto', 'quinze', 'relação', 'sabe', 'sabem', 'são', 'se', 'segunda', 'segundo', 'sei', 'seis', 'seja', 'sejam', 'sejamos', 'sem', 'sempre', 'sendo', 'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'sete', 'sétima', 'sétimo', 'seu', 'seus', 'sexta', 'sexto', 'si', 'sido', 'sim', 'sistema', 'só', 'sob', 'sobre', 'sois', 'somos', 'sou', 'sua', 'suas', 'tal', 'talvez', 'também', 'tampouco', 'tanta', 'tantas', 'tanto', 'tão', 'tarde', 'te', 'tem', 'tém', 'têm', 'temos', 'tendes', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'tens', 'ter', 'terá', 'terão', 'terceira', 'terceiro', 'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tiveste', 'tivestes', 'toda', 'todas', 'todavia', 'todo', 'todos', 'trabalho', 'três', 'treze', 'tu', 'tua', 'tuas', 'tudo', 'última', 'últimas', 'último', 'últimos', 'um', 'uma', 'umas', 'uns', 'vai', 'vais', 'vão', 'vários', 'vem', 'vêm', 'vendo', 'vens', 'ver', 'vez', 'vezes', 'viagem', 'vindo', 'vinte', 'vir', 'você', 'vocês', 'vos', 'vós', 'vossa', 'vossas', 'vosso', 'vossos', 'zero', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '_' ]
len(new_stopwords)

sem_acentuacao_stopwords = ['de','a','o','que','e','do','da','em','um','para','e','com','uma','os','no','se','na','por','mais','as','dos','como','mas','foi','ao','ele','das','tem','a','seu','sua','ou','ser','quando','muito','ha','nos','ja','esta','eu','tambem','so','pelo','pela','ate','isso','ela','entre','era','depois','sem','mesmo','aos','ter','seus','quem','nas','me','esse','eles','estao','voce','tinha','foram','essa','num','nem','suas','meu','as','minha','tem','numa','pelos','elas','havia','seja','qual','sera','nos','tenho','lhe','deles','essas','esses','pelas','este','fosse','dele','tu','te','voces','vos','lhes','meus','minhas','teu','tua','teus','tuas','nosso','nossa','nossos','nossas','dela','delas','esta','estes','estas','aquele','aquela','aqueles','aquelas','isto','aquilo','estou','esta','estamos','estao','estive','esteve','estivemos','estiveram','estava','estavamos','estavam','estivera','estiveramos','esteja','estejamos','estejam','estivesse','estivessemos','estivessem','estiver','estivermos','estiverem','hei','ha','havemos','hao','houve','houvemos','houveram','houvera','houveramos','haja','hajamos','hajam','houvesse','houvessemos','houvessem','houver','houvermos','houverem','houverei','houvera','houveremos','houverao','houveria','houveriamos','houveriam','sou','somos','sao','era','eramos','eram','fui','foi','fomos','foram','fora','foramos','seja','sejamos','sejam','fosse','fossemos','fossem','for','formos','forem','serei','sera','seremos','serao','seria','seriamos','seriam','tenho','tem','temos','tem','tinha','tinhamos','tinham','tive','teve','tivemos','tiveram','tivera','tiveramos','tenha','tenhamos','tenham','tivesse','tivessemos','tivessem','tiver','tivermos','tiverem','terei','tera','teremos','terao','teria','teriamos','teriam']
len(sem_acentuacao_stopwords)

def merge_stopwords(arr1, arr2):
    merged = arr1.copy()  # Cria uma cópia do primeiro array
    for element in arr2:
        if element not in merged:
            merged.append(element)  # Adiciona apenas os elementos que não estão presentes no primeiro array
    return merged

stopwords = merge_stopwords(stopwords, new_stopwords)

stopwords = merge_stopwords(stopwords, sem_acentuacao_stopwords)

stopwords.remove('não')

len(stopwords)

##### Função

def remove_stopwords(tokens):
  filtered_tokens = []
  for sentence in tokens:
      filtered = [palavra for palavra in sentence if palavra not in stopwords]
      filtered_tokens.append(filtered)
  return filtered_tokens

#### 4.6 Lematização


def lematizacao(tokens):
  # Carregar o modelo pré-treinado do SpaCy para o idioma português
  nlp = spacy.load("pt_core_news_sm")
  lemmatized_tokens = []

  for sentence in tokens:
    lemma_list = []
    doc = nlp(" ".join(sentence))  # Unir as palavras da frase em uma única string

    for token in doc:
      if token.lemma_ != '-PRON-':
        if token.pos_ == 'VERB':
          palavra_lematizada = token.lemma_
        else:
          palavra_lematizada = token.lemma_

        if palavra_lematizada:
          lemma_list.append(palavra_lematizada)

    lemmatized_tokens.append(lemma_list)
    
  # Converter todas as palavras para minúsculas
  lemmatized_tokens_lower = []
  for sentence in lemmatized_tokens:
    sentence_lower = [palavra.lower() for palavra in sentence]
    lemmatized_tokens_lower.append(sentence_lower)
  
  return lemmatized_tokens_lower

def pipeline(comment):
      # Tokenização
      tokens = tokenizer(comment)
      # Tratamento de Emojis
      #demojized = demojize_tokens(tokens)
      # Remoção dos alfanuméricos
      no_alfanumericos = removendo_alfanumericos(tokens)
      # Normalização das abreviações
      normalizado = comentarios_normalizados(no_alfanumericos, dicionario_girias)
      # Remoção das stopwords
      no_stopwords = remove_stopwords(normalizado)
      # lematização
      tratados = lematizacao(no_stopwords)
      
      return tratados


# Carrega o modelo treinado
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Classifica as linhas do CSV
def classify_rows(model, df):
    df_classificado = df
    df_classificado.insert(loc=1, column='classificacao', value=model.predict(df_classificado.iloc[:,1:50]))
    return df_classificado

def sentiment_chart(df2):
    # Contar os tipos de sentimento
    count_sentiment = df2['classificacao'].value_counts()
    # Criar o gráfico de pizza
    plt.figure(figsize=(2, 2), facecolor='#11111E')
    count_sentiment.plot(kind='pie', autopct='%1.1f%%', textprops={'color': 'white'})
    plt.ylabel('')
    st.pyplot(plt)

def first_process(df):
    df = df.rename(columns={'"anomalia"' : 'anomalia', '"dataPublicada"' : 'dataPublicada', '"autor"' : 'autor', '"texto"' : 'texto', '"sentimento"' : 'sentimento', '"tipoInteracao"' : 'tipoInteracao', '"probabilidadeAnomalia"' : 'probabilidadeAnomalia', '"linkPost"' : 'linkPost', '"processado"' : 'processado',  '"contemHyperlink"' : 'contemHyperlink' })
    df = df[df['anomalia'] != 1]
    df = df.drop(['id', 'dataPublicada', 'anomalia', 'probabilidadeAnomalia', 'linkPost', 'processado', 'contemHyperlink'], axis=1)
    df = df.loc[df['autor'] != 'btgpactual']
    df = df[df['autor'] != 'moinho_cultural']
    df = df.reset_index(drop=True)
    return df

def create_sentence_vector(model, df):
    sentence_table = []
    for sentence in df['texto']:
        word_vectors = [model[word] for word in sentence if word in model]
        if len(word_vectors) > 0:
            sentence_vector = sum(word_vectors) / len(word_vectors)
        else:
            sentence_vector = [None] * 100  # Cria uma lista de 100 elementos None
        sentence_table.append((sentence, *sentence_vector[:50]))  # Adiciona apenas os primeiros 50 elementos do vetor

    column_labels = ['Frase']
    for i in range(50):
        column_labels.append(f'Vetor{i+1}')
    df_vec = pd.DataFrame(sentence_table, columns=column_labels)

    df["sentimentoNumerico"] = df["sentimento"].replace({'NEGATIVE': -1, 'POSITIVE': 1, 'NEUTRAL': 0})

    # Definir o índice do DataFrame df_vec como o mesmo índice de df_processada['sentimentoNumerico']
    df_vec.set_index(df["sentimentoNumerico"].index, inplace=True)

    df_vec['sentimento'] = df["sentimentoNumerico"]
    df_vec = df_vec.dropna()

    return df_vec

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
        st.subheader(".CSV que você carregou:")
        st.dataframe(df.head(5), use_container_width=True)

        st.write("---")

        df2 = first_process(df)

        output = pipeline(df2['texto'])
        df3 = pd.DataFrame({'texto': output, 'sentimento': df2['sentimento']})

        st.subheader(".CSV processado:")
        st.dataframe(df3.head(5), use_container_width=True)

        st.write("---")

        cbow = "cbow_s50.txt"
        model_cbow = KeyedVectors.load_word2vec_format(cbow)
        df_vec = create_sentence_vector(model_cbow, df3)

        st.subheader(".CSV com vetorização:")
        st.dataframe(df_vec.head(5), use_container_width=True)

    
        # Carrega o modelo
        model_path = "naivebayes_word2vec_cbow_sprint3.pkl"  # Caminho para o arquivo .pkl do modelo treinado
        model = load_model(model_path)
        
        st.write("---")

        st.subheader("Resultado da classificação:")

        # Classifica as linhas
        df_final = classify_rows(model, df_vec)

        # separa a interface em duas colunas
        columns = st.columns((2,1))

        with columns[0]:
            st.subheader(".CSV classificado:")
            st.dataframe(df_final.iloc[:6,0:2], use_container_width=True)    

        with columns[1]:
            st.subheader("Palavras mais frequentes:")
            st.write(df_final.columns)
            
        with st.container():
            sentiment_chart(df_final)
    

# Executa o aplicativo
if __name__ == '__main__':
    main()
