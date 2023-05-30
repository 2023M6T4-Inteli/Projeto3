# **8. Vetorização**

## 8.1 Bag of Words

### 8.1.1 Introdução

&emsp;&emsp; O modelo _Bag of Words_ é uma das várias ferramentas de vetorização de frases e palavras, processo que é de suma importância para o desenvolvimento de um modelo PLN (Processamento de Linguagem Natural), visto que o modelo de _machine learning_ só pode receber números como _inputs_. 

### 8.1.2 Método
&emsp;&emsp; Como última etapa de manipulação de dados antes do uso do modelo de _Machine Learning_ para a classificação de resultados temos a vetorização dos comentários, processo que nessa _pipeline_ foi conduzido pelo modelo _Bag of Words (BoW)_. O modelo BoW consiste na elaboração de uma matriz a partir de um vocabulário de todos os vocábulos presentes nos textos, enquanto que cada linha será um comentário que se deseja vetorizar. É importante notar que esse modelo é menos robusto, considerando apenas a frequência de palavras em cada frase e não os sentidos semânticos. <br>
&emsp;&emsp; Para essa etapa, foi utilizada uma instância da classe `CountVectorizer()`, e seus métodos, da biblioteca _sklearn_ (scikit-learn) a fim de que fosse gerado um vocabulário e as respectivas correspondências para cada comentário.

### 8.1.3 Resultados

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/bow.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 27: Demonstração do Bag of Words
<br>

&emsp;&emsp; Após o corpus  dos textos terem passado pelo _pipeline_, chega o momento de analisar as repetições de acordo com cada comentário feito, por meio da técnica _Bag of Words (BoW)_ utilizada em processamento de linguagem natural (PLN). Essa técnica é utilizada para representar um texto como um conjunto de palavras desordenadas, ignorando a ordem e a estrutura gramatical das frases.  Nesse modelo, cada palavra única do texto é transformada em uma _feature_ (característica), e a frequência de cada palavra no texto é usada como um valor numérico para a _feature_ correspondente.
<br>
&emsp;&emsp; Por exemplo, a frase "O gato preto pulou o muro" seria representada como um conjunto de palavras desordenadas: `'o', 'gato', 'preto', 'pulou', 'o', 'muro'`. A frequência de cada palavra é contada, e o resultado é um vetor numérico que representa a frequência de cada palavra na frase. O modelo _Bag of Words_ é uma técnica simples e eficiente para representar textos em formato vetorial, o que permite utilizá-los em algoritmos de aprendizado de máquina. 
<br>
&emsp;&emsp; Assim, abaixo é possível visualizar o código necessário para realizar essa vetorização e o _output_ dele:
```
def bow(comentarios): 
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    bow_model = vectorizer.fit_transform(comentarios)
    bow_df = pd.DataFrame(bow_model.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df
```
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/output.jpg"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 28: Output do código
<br>

&emsp;&emsp; Abaixo é demonstrado um exemplo resultante desta tabela, a qual possui um total de 12.193 linhas, que estão de acordo com cada comentário do csv disponibilizado pelo cliente, além de 24.331 colunas, que foram as palavras chaves selecionadas.
```
df['conf'].value_counts() 
0    11795
1      396
2        2
Name: conf, dtype: int64
```
&emsp;&emsp; Neste exemplo, é possível perceber que o termo `‘conf’`  se repete uma vez, em 396 comentários diferentes, e se repete duas vezes em 2 comentários diferentes. Dessa forma, percebe-se como a função consegue selecionar palavras chaves que estão contidas nas diversas frases do dataframe.

### 8.1.4 Conclusão
&emsp;&emsp; Com a aplicação do Modelo _Bag of Words (BoW)_ é possível perceber a capacidade de seleção de palavras para a futura implementação na _Machine Learning_ desenvolvida. O objetivo do projeto é demonstrado a partir da imagem abaixo:

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/modelo.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 29: Demonstração do modelo pronto
<br>

&emsp;&emsp; Porém, foi possível analisar que é necessário uma renovação no tratamento dos dados e exclusão de determinadas palavras, já que foi percebido que havia uma alta diversidade de termos que estão exclusos e/ou outros que permanecerão nas frases e não deveriam permanecer. Abaixo há exemplo desta análise:

```
word_counts = df.sum()
top_words = word_counts.sort_values(ascending=False)
top_10 = top_words.head(10)
top_10

btgpactual    6489
invest        4014
btg           2822
tod           1783
banc          1771
sobr          1364
melhor        1363
cont          1332
merc          1305
financeir     1303
dtype: int64
```

&emsp;&emsp; Além disso, foi feita uma _plotagem_ de uma nuvem de palavras para ser mais intuitiva a visualização dos termos que serão necessários passar por um tratamento.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/nuvem_palavras.png"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 30: Nuvem de palavras
<br>

&emsp;&emsp; Assim, o próximo passo é um retratamento dos textos para ter melhor desenvolvimento e aplicação no momento de construção da Inteligência Artificial.


## 8.2 Word2Vec

### 8.2.1 Introdução
&emsp;&emsp; O modelo de vetorização Word2Vec, ao contrário do Bag Of Words, permite um entendimento das conexões semânticas das palavras, sendo assim, palavras podem ser analisadas a partir de sua similaridade com outras através da ocorrência contextual. Isso é possível porque o Word2Vec relaciona as palavras a um espaço dimensional, representadas por um vetor denso, onde palavras com similaridade maior tendem a se agrupar no mesmo espaço. O modelo, através do uso de uma rede neural embutida, aprende por treinamento em big corpus, como o wikipédia, a localização ideal para cada palavra.

### 8.2.2 Método
&emsp;&emsp;No projeto recorremos a 2 métodos de vetorização com o Word2Vec: o modelo pré-treinado de Continuous Bag-of-Words (CBOW) da biblioteca NILC e, posteriormente, o treinado no nosso próprio corpus. O modelo pré-treinado, como dito anteriormente, não necessita de treinamento adicional, pois ele já dominou as relações linguísticas entre as palavras a partir do primeiro treinamento, além disso, o modelo usado é público e permite o uso quase instantâneo no corpus. 

### 8.2.3 Resultados
&emsp;&emsp;Para ambos os métodos, foi necessário realizar a soma dos vetores de palavras presentes nas frases, desse modo, teremos um vetor para cada frase. No método pré-treinado, construímos vetores de 50 dimensões, enquanto que, no modelo treinado no corpus da base de dados de comentários, construímos vetores de 100 dimensões.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/word2vec_pre_treinado.jpg"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 31: Word2Vec com modelo pré-treinado
<br>

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/word2vec_corpus.jpg"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 32: Word2Vec treinado com o corpus
<br>

### 8.2.4 Conclusão
&emsp;&emsp;A conclusão obtida de qual será o melhor método de vetorização se dará na construção dos modelos de machine learning que será explicado na próxima sessão, pois assim poderemos comparar diretamente a acurácia e recall dos diferentes modelos.



