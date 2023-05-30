# **9. Modelos**

## 9.1 Word2Vec com CBOW

### 9.1.1 Introdução

&emsp;&emsp; O Word2Vec é um modelo de aprendizado de representação de palavras, que captura eficientemente as relações semânticas entre palavras com base em seu contexto. Nesse caso é utilizado esse modelo com o CBOW (Continuous Bag-of-Words), que já é um modelo treinado.

### 9.1.2 Método
&emsp;&emsp; Nesse caso, como é um modelo pré-treinado, o único processo que foi necessário foi o Label Encoding, onde é transformado a coluna ‘sentimento’ em valores numéricos: -1, 0 e 1. O output é uma tabela com: 1 coluna com a frase do post, 50 colunas de vetores e 1 coluna de sentimento.

### 9.1.3 Resultados
&emsp;&emsp; A primeira etapa da realização do Word2Vec com CBOW é ler o arquivo do modelo já treinado.

```
cbow = 'caminho_do_arquivo'
model_cbow = KeyedVectors.load_word2vec_format(cbow)
```

&emsp;&emsp;Para fins acadêmicos, o modelo acima foi testado com duas palavras para que se pudesse visualizar os vetores delas e provar que funciona. Abaixo está o output encontrado. 

```
wordvec_test = model_cbow['projeto']
wordvec_test

	output: array([-0.074174, -0.152088,  0.086627, -0.224567,  0.362562,  0.130683, -0.089179, -0.086973,  0.309501,  0.004112, -0.308202,  0.351789, -0.477863,  0.050276,  0.213283,  0.159895, -0.285545, -0.08832 , -0.015449,  0.014816, -0.613861,  0.502556,  0.021688,  0.369492, 0.280691,  0.016868,  0.105584, -0.180754, -0.078456,  0.148032, 0.36293 , -0.011634,  0.412191, -0.009049,  0.010404,  0.131242, -0.032483, -0.133067, -0.063802,  0.434015, -0.214768, -0.072132, 0.045601, -0.368866,  0.502808,  0.048293, -0.254894,  0.142581, -0.075066,  0.015646], dtype=float32)

wordvec_test = model_cbow['banco']
wordvec_test
	array([ 1.81041e-01,  1.07700e-01, -1.04667e-01,  2.43361e-01,
        6.06380e-02,  3.92829e-01, -3.33944e-01, -3.81778e-01,
        1.42200e-01,  8.59360e-02, -1.16615e-01,  3.95722e-01,
       -6.12684e-01, -7.68980e-02,  3.34396e-01,  8.11270e-02,
       -5.17700e-02, -3.21950e-01, -6.91509e-01, -3.31210e-01,
       -5.43213e-01,  6.09881e-01,  2.43700e-01,  3.73240e-02,
        1.16518e-01,  1.78859e-01, -3.78839e-01,  1.27430e-01,
        1.94497e-01,  7.32000e-04,  3.14395e-01, -2.04550e-01,
        5.34431e-01, -5.55100e-03,  3.52343e-01, -4.92000e-02,
       -1.38384e-01,  2.31630e-02, -3.40013e-01,  5.00201e-01,
       -1.14170e-02, -1.29925e-01, -6.12800e-03, -1.80481e-01,
        1.99391e-01,  1.37645e-01, -7.66434e-01, -2.26784e-01,
       -6.16110e-02,  9.05920e-02], dtype=float32)

```
&emsp;&emsp;	A função abaixo, chamada de vetorizando() recebe um modelo de vetores de palavras treinado e um dataFrame (df) contendo um texto já pré tratado. Ela verifica se a palavra na frase está no modelo de vetores, caso tenha, o vetor é adicionado a uma lista. A seguir, a função calcula o vetor médio dessas palavras encontradas na sentença, caso a palavra não seja encontrada, é criada uma lista de 100 elementos "None". A função armazena a sentença original e os primeiros 50 elementos do vetor médio, além de criar o df_vec, uma nova base de dados. <br>
&emsp;&emsp;	O dataframe original df é modificado adicionando uma coluna sentimentoNumerico que transforma as categorias de sentimento: "NEGATIVE", "POSITIVE" e "NEUTRAL", para valores numéricos: -1, 1 e 0, respectivamente. Em seguida, a função adiciona a coluna sentimento a df_vec com base em df['sentimentoNumerico']. Em seguida, ela remove quaisquer linhas que contenham valores ausentes no df_vec e retorna o dataframe.

```
def create_sentence_vector(model, df):
    sentence_table = []
    for sentence in df['texto_tratado']:
        word_vectors = [model[word] for word in sentence if word in model]
        if len(word_vectors) > 0:
            sentence_vector = sum(word_vectors) / len(word_vectors)
        else:
            sentence_vector = [None] * 100  
        sentence_table.append((sentence, *sentence_vector[:50])) 
    column_labels = ['Frase']
    for i in range(50):
        column_labels.append(f'Vetor{i+1}')
    df_vec = pd.DataFrame(sentence_table, columns=column_labels)
    df["sentimentoNumerico"] = df["sentimento"].replace({'NEGATIVE': -1, 'POSITIVE': 1, 'NEUTRAL': 0})
    df_vec.set_index(df["sentimentoNumerico"].index, inplace=True)
    df_vec['sentimento'] = df["sentimentoNumerico"]
    df_vec = df_vec.dropna()

    return df_vec
```

&emsp;&emsp;O código abaixo testa a função definida e tem como output a imagem abaixo. 

```
df_vec = create_sentence_vector(model_cbow, df)
	df_vec
```

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/word2vec_pre_treinado.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 33: Output do Word2Vec + CBOW
<br>

### 9.1.4 Conclusão
&emsp;&emsp; Em conclusão, o método Word2Vec, em particular com o CBOW, é uma técnica eficaz para a representação de palavras em espaços vetoriais contínuos. O processo de vetorização de um texto usando o CBOW envolve a criação de vetores de palavras e o cálculo de vetores médios para as sentenças.


## 9.2 Naive Bayes + Word2Vec com CBOW

### 9.2.1 Introdução
&emsp;&emsp;O Naive Bayes é um classificador probabilístico amplamente utilizado em problemas de aprendizado de máquina. Ele se baseia no teorema de Bayes para estimar a probabilidade condicional das classes com base em evidências fornecidas pelas características dos dados. Por outro lado, o Word2Vec com CBOW é um algoritmo de aprendizado de representação de palavras que visa capturar as relações semânticas e sintáticas entre as palavras em um corpus de texto.

### 9.2.2 Método
&emsp;&emsp;Na abordagem que combina Naive Bayes e Word2Vec com CBOW, o Word2Vec é primeiro treinado em um corpus de texto já treinado para aprender as representações vetoriais das palavras. Em seguida, essas representações vetoriais são utilizadas como características no modelo Naive Bayes. Por isso, o dataframe referenciado neste tópico será o gerado no tópico anterior. 

### 9.2.3 Resultados
&emsp;&emsp;No exemplo a seguir, é considerado um conjunto de dados representado por um dataframe df_vec. A coluna sentimento representa os valores numéricos dos sentimentos associados a cada texto, enquanto as colunas restantes, de 1 a 50, contêm as representações vetoriais das palavras. O código abaixo define a variável-alvo que será usada no treinamento do modelo de classificação.

```
target = df_vec['sentimento']
```

&emsp;&emsp;Para selecionar os recursos relevantes para o modelo de classificação, é utilizado o código abaixo, que seleciona todas as linhas do dataframe e as colunas de 1 a 50, que correspondem às representações vetoriais das palavras. A figura a seguir demonstra o output. 

```
feature = df_vec.iloc[:,1:50]
feature
```

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/naivebayes_word2vec_cbow.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 34: Output Naive Bayes + Word2Vec com CBOW
<br>

&emsp;&emsp;No código abaixo, é realizado um particionamento dos dados em conjuntos de treinamento e teste usando a função train_test_split, ela divide um conjunto de dados em subconjuntos para fins de treinamento e avaliação de modelos de aprendizado de máquina.

```
	X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
```

&emsp;&emsp; A primeira linha cria um objeto do tipo GaussianNB, que é o classificador Naive Bayes Gaussiano. Na segunda linha o modelo é treinado usando o método fit(), o conjunto de treinamento X_train, já definido, é fornecido como as características e y_train.values.ravel() como a variável-alvo. O método values.ravel() é utilizado para converter y_train em um array unidimensional, necessário para o treinamento do modelo. A função predict() é usada para fazer previsões com base nos dados de teste e as previsões resultantes são armazenadas na variável Y_pred. Por fim, a função classification_report é utilizada para imprimir o relatório de classificação, comparando as previsões Y_pred com as verdadeiras classes do conjunto de teste y_test e exibe métricas como precisão, recall, F1-score e suporte para cada classe, imagem presente após o código abaixo.

```
clf = GaussianNB()
clf = clf.fit(X_train,y_train.values.ravel())
Y_pred = clf.predict(X_test)
print(classification_report(y_test, Y_pred))
```

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/output_naive_word2vec_cbow.jpeg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 	Figura 35: Output da métricas	
<br>

&emsp;&emsp;A função accuracy_score(y_test, Y_pred) abaixo é usada para comparar as previsões feitas pelo modelo (Y_pred) com as verdadeiras classes do conjunto de teste (y_test), definindo à variável acc_score. Para exibir o resultado de forma mais legível, a porcentagem da precisão é formatada com a função format() do Python. A expressão "{:.2%}".format(acc_score) indica que a formatação com 2 casas decimais e o símbolo de porcentagem.

```
acc_score = accuracy_score(y_test, Y_pred)
format_output = "{:.2%}".format(acc_score)
print("Precisão final de :",format_output)

output: Precisão final de : 43.35%
```

### 9.2.4 Conclusão
&emsp;&emsp;O modelo Word2Vec com CBOW é capaz de capturar informações contextuais e semânticas das palavras, fornecendo representações vetoriais que preservam relações entre as palavras. Essas representações vetoriais são úteis para entender a semelhança e a estrutura do texto, enriquecendo a qualidade das características utilizadas pelo Naive Bayes.


## 9.3 Word2Vec com o corpus

### 9.3.1 Introdução
&emsp;&emsp; Uma das principais técnicas usadas no Word2Vec é a camada de embedding. Essa camada é responsável por mapear palavras individuais para vetores de números reais em um espaço de alta dimensão. Cada palavra é representada por um vetor denso, onde as dimensões desse vetor capturam informações contextuais e semânticas sobre a palavra.

### 9.3.2 Método
&emsp;&emsp; Nesse caso o Word2Vec é treinado diretamente no corpus do dataframe, ao invés de já ter um modelo pré treinado. Para realizar esse processo, é utilizado a biblioteca gensim.models, importando o Word2Vec.

### 9.3.3 Resultados
&emsp;&emsp; O código abaixo se inicia com a importação da classe Word2Vec da biblioteca Gensim, que é usada para treinar o modelo. Em seguida, a função train_word2vec() recebe dois parâmetros: df, que é o dataframe, e column_name, que é o nome da coluna que contém as frases tokenizadas. <br>
&emsp;&emsp; A variável sentences é inicializada com uma lista das frases tokenizadas presentes na coluna especificada. A função tolist() é usada para converter os valores da coluna em uma lista. <br>
&emsp;&emsp; Em seguida, o modelo é treinado usando a função Word2Vec do Gensim. O parâmetro sentences é passado para representar o corpus de treinamento, já o parâmetro min_count=1 indica que todas as palavras devem ter pelo menos uma ocorrência para serem consideradas no treinamento. E por fim, o modelo é retornado. . <br>

```
from gensim.models import Word2Vec

def train_word2vec(df, column_name):
    sentences = df[column_name].tolist()
    model = Word2Vec(sentences, min_count=1)
    return model
```

&emsp;&emsp; A primeira função abaixo recebe dois parâmetros: model, que é o modelo Word2Vec treinado, e sentence, que é uma lista de palavras tokenizadas representando uma frase. A seguir, é inicializada uma lista vazia chamada vectors para armazenar os vetores de palavras. <br>
&emsp;&emsp; Em seguida, ocorre um loop sobre cada palavra na lista sentence. A propriedade model.wv verifica se a palavra está presente no vocabulário do modelo Word2Vec, e caso esteja presente, o seu vetor é obtido usando model.wv[word] e adicionado à lista vectors. <br>
&emsp;&emsp; Depois de iterar todas as palavras da frase, é feita uma verificação se a lista vectors contém algum vetor, e se houver, eles são somados usando np.sum(vectors, axis=0) para obter um vetor que representa a frase como um todo. Esse vetor é então normalizado dividindo-o pelo número de palavras na frase (len(sentence)) para obter a média dos vetores. Porém, caso a lista vectors esteja vazia, é retornado um vetor de zeros com o mesmo tamanho dos vetores do modelo (np.zeros(model.vector_size)). <br>

```
def get_word_vectors(model, sentence):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word]) # Append na lista de vetores
    if vectors:
        return np.sum(vectors, axis=0)/len(sentence) # Soma dos vetores para cada frase
    else:
        return np.zeros(model.vector_size)
```

&emsp;&emsp; A segunda função recebe três parâmetros: df, que é um dataframe contendo os dados, column_name, que é o nome da coluna, e model, que é o modelo Word2Vec treinado. Primeiramente, a função converte as frases tokenizadas da coluna especificada em uma lista chamada sentences, e em seguida, ocorre um loop sobre cada frase na lista. Para cada frase, a função get_word_vectors é chamada para obter o vetor representativo da frase. <br>

&emsp;&emsp; Os vetores resultantes para cada frase são armazenados na lista vectors, que é construída utilizando uma compreensão de lista, onde cada elemento da lista é o vetor. Após iterar por todas as frases, é criado um novo dataframe chamado df_vectors, onde cada coluna representa uma dimensão do vetor. O número de colunas é determinado pelo tamanho dos vetores do modelo (model.vector_size). O df_vectors é então concatenado ao dataframe original df usando a função pd.concat, resultando no dataframe final df_word2vec.

```
def create_word2vec_dataframe(df, column_name, model):
    sentences = df[column_name].tolist()
    vectors = [get_word_vectors(model, sentence) for sentence in sentences]
    df_vectors = pd.DataFrame(vectors, columns=[f"Vetor{i}" for i in range(model.vector_size)])
    df_word2vec = pd.concat([df, df_vectors], axis=1)
    return df_word2vec
```


&emsp;&emsp;Por fim, as funções são testadas no dataframe original.


### 9.3.4 Conclusão
&emsp;&emsp;O código apresentado ilustra como treinar um modelo Word2Vec usando a biblioteca Gensim e como criar um dataframe com vetores de palavras para frases tokenizadas usando esse modelo. E o Word2Vec com a camada de embedding é uma abordagem poderosa para aprender representações vetoriais de palavras em tarefas de processamento de linguagem natural.



## 9.4 Naive Bayes + Word2Vec com o corpus

### 9.4.1 Introdução
&emsp;&emsp; Da mesma forma que o Naive Bayes funciona com o Word2Vec + CBOW é o jeito que funciona com esse modelo, o Word2Vec com o corpus. A grande diferença é que o segundo não utiliza um modelo já pré treinado, o que pode ou não melhorar o resultado final.

### 9.4.2 Método
&emsp;&emsp; Na abordagem que combina Naive Bayes e Word2Vec com o corpus, o Word2Vec é treinado com o corpus do dataset. Em seguida, essas representações vetoriais são utilizadas como características no modelo Naive Bayes. 

### 9.4.3 Resultados
&emsp;&emsp; A primeira etapa a ser feita é a separação entre teste e treino, como mostra os códigos abaixo. O primeiro, cria a variável target, que armazena a coluna chamada sentimentoNumerico do df_word2vec, onde cada valor nessa coluna representa o sentimento atribuído. Essa variável target será usada posteriormente como o objetivo de indicar qual sentimento é esperado para cada instância do corpus.

```
target = df_word2vec['sentimentoNumerico']
```

&emsp;&emsp; O segundo, cria a variável feature, que contém um recorte do df_word2vec. Mais especificamente, todas as linhas e todas as colunas da posição 2 até a posição 101 são selecionadas. Cada valor dessa variável feature representa um componente do vetor Word2Vec associado a uma palavra específica do texto. 

```
feature = df_word2vec.iloc[:,2:102]
```

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/word2vec_corpus.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 	Figura 36: Output feature	
<br>

&emsp;&emsp; Os códigos a seguir são os mesmos utilizados anteriormente para a separação de treino e teste no Naive Bayes e avaliação do modelo, e a seguir há uma imagem com os resultados obtidos.  

```
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)


clf = GaussianNB()
clf = clf.fit(X_train,y_train.values.ravel())
Y_pred = clf.predict(X_test)
print(classification_report(y_test, Y_pred))

```

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/output_naive_word2vec.jpeg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 37: Output das métricas	
<br>


### 9.4.4 Conclusão
&emsp;&emsp; 	Em conclusão, o código apresentado realiza a preparação dos dados e a divisão do conjunto de características e rótulos para aplicação do método Naive Bayes com o Word2Vec. O objetivo desse código é realizar a classificação ou análise de texto com base nos vetores Word2Vec gerados a partir do corpus.


