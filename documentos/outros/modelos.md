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


## 9.5 Naive Bayes + BoW

### 9.5.1 Introdução

&emsp;&emsp; Esse modelo junta o algoritmo Naive Bayes com um modelo de vetorização, já explicado, que é o BoW. Se espera que, com essa junção, os resultados sejam promissores nesta tarefa de classificação de texto. Todos os modelos que serão apresentados foram rodados com a base de dados com o pré-processamento da Sprint 3 e Sprint 4, por isso serão apresentados os resultados dos dois dataframes.

### 9.5.2 Método

#### 9.5.2.1 Cross Validation
&emsp;&emsp; Cross validation é uma técnica usada para avaliar a capacidade de um modelo de generalizar para novos dados, que consiste em dividir o conjunto de dados em partes menores, treinar o modelo em uma parte e testá-lo em outra. Esse processo é repetido várias vezes e a média das métricas de avaliação é usada para avaliar o desempenho do modelo.

#### 9.5.2.2 Grid search
&emsp;&emsp; Grid search é uma técnica de busca de hiperparâmetros usada para encontrar a melhor combinação de valores para um modelo de aprendizado de máquina, que consiste em definir um conjunto de valores para cada hiperparâmetro e treinar e avaliar o modelo com todas as combinações possíveis. O conjunto de hiperparâmetros que produz a melhor métrica de avaliação é selecionado como a configuração final do modelo.

### 9.5.3 Resultados

#### 9.5.3.1 Naive Bayes
&emsp;&emsp; Primeiramente, o objeto LabelEncoder() é criado, a fim de transformar as classes de texto em números inteiros, já que o Naive Bayes somente trabalha com valores numéricos. O encoder é ajustado nos dados da coluna sentimento do df, e as classes são transformadas em números inteiros usando o método fit_transform, armazenando-os na variável sentimento. Em seguida, os dados são divididos em conjuntos de treino e teste usando a função train_test_split, os conjuntos de treino (X_treino e y_treino) e teste (X_teste e y_teste) são criados a partir dos dados do bow_model e dos rótulos transformados, com 20% dos dados destinados ao conjunto de teste e o restante para o conjunto de treino.

```
encoder = LabelEncoder() 

sentimento = encoder.fit_transform(df['sentimento'])

X_treino, X_teste, y_treino, y_teste = train_test_split(bow_model, sentimento, test_size=0.2, random_state=42)
```
&emsp;&emsp; Após a divisão dos dados, o objeto do tipo MultinomialNB() é criado, e  o modelo é treinado usando o conjunto de treino através do método fit, passando as matrizes de treino e os rótulos correspondentes. A próxima etapa é fazer a predição usando os dados de teste, por isso o método predict é aplicado ao modelo treinado usando os dados de teste, gerando previsões numéricas para as classes. Estas são decodificadas para obter as classes originais usando o método inverse_transform e são armazenadas na variável predicao. Por fim, é impresso o relatório de classificação.

```
modelo = MultinomialNB()
modelo.fit(X_treino, y_treino)

predicao_numerica = modelo.predict(X_teste)

predicao = encoder.inverse_transform(predicao_numerica)

print(classification_report(df['sentimento'].iloc[y_teste], predicao))
```

&emsp;&emsp; Abaixo é possível observar o código necessário para criar a matriz de confusão, onde esta é calculada usando a função confusion_matrix(df['sentimento'].iloc[y_teste], predicao), que recebe como parâmetros os rótulos verdadeiros (df['sentimento'].iloc[y_teste]) e as previsões feitas pelo modelo (predicao). Após calcular a matriz de confusão, uma lista chamada classes é definida, contendo os nomes das classes presentes no problema, que serão utilizados para rotular as classes na matriz de confusão. Em seguida, é criada uma figura de plotagem com as dimensões especificadas usando plt.figure(figsize=(8, 6)). A função sns.heatmap() é chamada para criar o mapa de calor, os parâmetros são: cm é que é a matriz de confusão, annot=True e fmt='g' utilizados para exibir os valores da matriz nas células, cmap='Blues' que define a paleta de cores a ser utilizada no mapa de calor, e por último, os rótulos dos eixos x e y são definidos com base na lista de classes usando os parâmetros xticklabels e yticklabels.

```
cm = confusion_matrix(df['sentimento'].iloc[y_teste], predicao)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Relatório de Classificação - Sprint 3** <br>
```
             precision    recall  f1-score   support
           0       0.00      0.00      0.00         0
           1       0.88      0.46      0.61      1230
           2       0.63      0.76      0.69       612

    accuracy                           0.56      1842
   macro avg       0.50      0.41      0.43      1842
weighted avg       0.79      0.56      0.63      1842
```
&emsp;&emsp; **Matriz de Confusão - Sprint 3** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_naiveBayes_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 36: Matriz de confusão Naive Bayes - Sprint 3	
<br>
&emsp;&emsp; **Relatório de Classificação - Sprint 4** <br>
```
             precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       0.85      0.30      0.45       957
           2       0.66      0.79      0.72       651

    accuracy                           0.50      1608
   macro avg       0.50      0.36      0.39      1608
weighted avg       0.77      0.50      0.56      1608
```
&emsp;&emsp; **Matriz de Confusão - Sprint 4** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_naiveBayes_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 37: Matriz de confusão Naive Bayes - Sprint 4	
<br>

#### 9.5.3.2 Naive Bayes - Cross Validation
&emsp;&emsp; Na primeira linha, o código utiliza a função cross_val_score para calcular as pontuações de Cross Validation do modelo, este é avaliado utilizando o conjunto de features bow_model e os rótulos de classe sentimento, o parâmetro cv=7 especifica que a cross validation será realizada em 7 folds. A seguir, o código imprime a média das pontuações obtidas, utilizando a função mean() no objeto scores, que representa o desempenho médio do modelo em todos os folds.

```
scores = cross_val_score(modelo, bow_model, sentimento, cv=7)

print('Acurácia média:', scores.mean())

output: 
Acurácia média: 0.6724311486588003 Sprint 3
Acurácia média: 0.6263681362502335 Sprint 4
```

&emsp;&emsp; Em seguida, o código utiliza a função cross_val_predict para fazer as previsões do modelo, os parâmetros chamados são os mesmos da função descrita no último parágrafo: modelo, bow_model, sentimento, cv=7. Na segunda linha, é utilizado o objeto encoder para decodificar as classes preditas (predicoes) de volta aos seus valores originais. Por fim, o código imprime o relatório de classificação usando a função classification_report. 

```
predicoes = cross_val_predict(modelo, bow_model, sentimento, cv=7)

predicao = encoder.inverse_transform(predicoes)

print('Relatório de Classificação:')
print(classification_report(df['sentimento'], predicao))
```

&emsp;&emsp;O código da Matriz de Confusão do Naive Bayes com Cross Validation é o mesmo do apresentado anteriormente.

&emsp;&emsp; **Relatório de Classificação - Sprint 3** <br>
```
             precision    recall  f1-score   support

           0       0.63      0.73      0.68      1974
           1       0.78      0.58      0.67      4012
           2       0.62      0.75      0.68      3221

    accuracy                           0.67      9207
   macro avg       0.67      0.69      0.67      9207
weighted avg       0.69      0.67      0.67      9207
```
&emsp;&emsp; **Matriz de Confusão - Sprint 3** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_naiveBayes_cross_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 38: Matriz de confusão Naive Bayes Cross Validation - Sprint 3	
<br>

&emsp;&emsp; **Relatório de Classificação - Sprint 4** <br>
```
             precision    recall  f1-score   support

           0       0.58      0.75      0.65      1970
           1       0.73      0.36      0.48      2918
           2       0.62      0.79      0.70      3152

    accuracy                           0.63      8040
   macro avg       0.64      0.64      0.61      8040
weighted avg       0.65      0.63      0.61      8040
```
&emsp;&emsp; **Matriz de Confusão - Sprint 4** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_naiveBayes_cross_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 39: Matriz de confusão Naive Bayes Cross Validation - Sprint 4	
<br>

#### 9.5.3.3 Naive Bayes - Grid Search
&emsp;&emsp; Na primeira linha, são definidos os valores a serem testados para os hiperparâmetros do modelo, e nesse caso, o hiperparâmetro alpha será testado com os valores [0.1, 0.5, 1.0, 2.0, 5.0], e o hiperparâmetro fit_prior será testado com os valores [True, False]. Em seguida, na terceira linha, uma instância do modelo Naive Bayes Multinomial é criada utilizando a classe MultinomialNB().

```
parametros = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0], 'fit_prior': [True, False]}

modelo = MultinomialNB()
```

&emsp;&emsp; A seguir, é criada uma instância do objeto GridSearchCV, que é utilizado para realizar uma busca exaustiva dos melhores hiperparâmetros para o modelo, esse objeto recebe o modelo criado, os parâmetros a serem testados, o número de folds para a validação cruzada (cv=5) e a métrica de avaliação a ser utilizada (scoring=”accuracy”). Na próxima linha, o modelo é treinado através do método fit() do objeto grid, e os dados de treino (X_treino e y_treino) são utilizados para treinar o modelo e encontrar os melhores hiperparâmetros. Por último, os melhores hiperparâmetros são exibidos pelo atributo best_params_ e a melhor acurácia com o atributo best_score_. 

```
grid = GridSearchCV(modelo, parametros, cv=5, scoring='accuracy')

grid.fit(X_treino, y_treino)

print('Melhores hiperparâmetros:', grid.best_params_)
print('Melhor acurácia:', grid.best_score_)

output:
Sprint 3:
Melhores hiperparâmetros: {'alpha': 0.5, 'fit_prior': True}
Melhor acurácia: 0.693550577053632
Sprint 4:
Melhores hiperparâmetros: {'alpha': 1.0, 'fit_prior': True}
Melhor acurácia: 0.6517396721129225
```

&emsp;&emsp; A seguir, um novo modelo é criado com os melhores hiperparâmetros encontrados. O alpha e o fit_prior são definidos de acordo com os valores ótimos encontrados pelo GridSearchCV, e esse novo modelo é treinado com os dados de treino. Após o treinamento, o modelo é utilizado para fazer a predição dos dados de teste (X_teste), que são retornadas como valores numéricos. Por meio do objeto encoder, as classes preditas numéricas são decodificadas.

```
modelo = MultinomialNB(alpha=grid.best_params_['alpha'], fit_prior=grid.best_params_['fit_prior'])
modelo.fit(X_treino, y_treino)

predicao_numerica = modelo.predict(X_teste)

predicao = encoder.inverse_transform(predicao_numerica)
```

&emsp;&emsp; Por fim, a função accuracy_score é utilizada, e nela são passados dois argumentos: o primeiro argumento, df['sentimento'].iloc[y_teste], refere-se às classes reais do conjunto de teste, e o segundo argumento é o predicao, que representa as classes preditas pelo modelo para o conjunto de teste. Por fim, o resultado da acurácia é exibido na tela utilizando a função print.

```
acuracia = accuracy_score(df['sentimento'].iloc[y_teste], predicao)
print('Acurácia no conjunto de teste:', acuracia)

Sprint 3:
Acurácia no conjunto de teste: 0.5537459283387622 
Sprint 4:
Acurácia no conjunto de teste: 0.4993781094527363 
```

&emsp;&emsp; O próximo código calcula a revocação do modelo descrito. Primeiramente, é definida uma grade de valores para os hiperparâmetros a serem testados, que são os mesmos definidos anteriormente, em seguida, é criada uma instância do modelo MultinomialNB.

```
parametros = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0], 'fit_prior': [True, False]}

modelo = MultinomialNB()
```

&emsp;&emsp; Após isso, é criada uma instância da métrica de avaliação 'recall', que é uma medida de desempenho que indica a proporção de instâncias positivas corretamente classificadas em relação ao total de instâncias positivas. Neste caso, a média 'macro' é utilizada, o que significa que o recall será calculado para cada classe individualmente e a média desses valores será obtida. Na próxima linha, é criado uma instância do objeto GridSearchCV, que realiza a busca exaustiva de hiperparâmetros através da validação cruzada. Ele recebe os três primeiros parâmetros iguais ao código anterior, e o últim há uma mudança de scoring, de accuracy para recall (scoring=recall).

```
parametros = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0], 'fit_prior': [True, False]}

modelo = MultinomialNB()

recall = make_scorer(recall_score, average='macro')

grid = GridSearchCV(modelo, parametros, cv=5, scoring=recall)
```

&emsp;&emsp; Por último, o modelo é treinado por meio do método fit, passando os dados de treinamento (X_treino) e os rótulos correspondentes (y_treino). E os resultados são exibidos pela função print.

```
grid.fit(X_treino, y_treino)

print('Melhores hiperparâmetros:', grid.best_params_)
print('Melhor revocação:', grid.best_score_)

output:
Sprint 3:
Melhores hiperparâmetros: {'alpha': 0.1, 'fit_prior': True}
Melhor revocação: 0.7103233123310607
Sprint 4:
Melhores hiperparâmetros: {'alpha': 0.5, 'fit_prior': True}
Melhor revocação: 0.6609166980005126
```

### 9.5.4 Conclusão
&emsp;&emsp;Pode-se concluir que esse modelo teve resultados satisfatórios para o projeto, e que o grupo pode tirar diversos insights. Além disso, pode-se perceber que o pré processamento feito na Sprint 4 não obteve um melhor sucesso, comparado com a Sprint 3, nesse modelo.

## 9.6 Random Forest + BoW

### 9.6.1 Introdução

&emsp;&emsp; Foi criado um modelo de Random Forest juntamente do processo de vetorização Bag of Words. Esse modelo é um algoritmo de aprendizado de máquina que combina várias árvores de decisão para formar um modelo mais preciso e robusto, criando várias árvores de decisão independentes, onde cada árvore é treinada em uma amostra aleatória do conjunto de dados e um subconjunto aleatório das características. Para fins de comparação, foram desenvolvidos códigos que têm a base de dados diferentes. 

### 9.6.2 Método

#### 9.6.2.1 Cross Validation
&emsp;&emsp; Cross validation é uma técnica usada para avaliar a capacidade de um modelo de generalizar para novos dados, que consiste em dividir o conjunto de dados em partes menores, treinar o modelo em uma parte e testá-lo em outra. Esse processo é repetido várias vezes e a média das métricas de avaliação é usada para avaliar o desempenho do modelo.

#### 9.6.2.2 Grid search
&emsp;&emsp; Grid search é uma técnica de busca de hiperparâmetros usada para encontrar a melhor combinação de valores para um modelo de aprendizado de máquina, que consiste em definir um conjunto de valores para cada hiperparâmetro e treinar e avaliar o modelo com todas as combinações possíveis. O conjunto de hiperparâmetros que produz a melhor métrica de avaliação é selecionado como a configuração final do modelo.

### 9.6.3 Resultados

#### 9.6.3.1 Random Forest

&emsp;&emsp; Primeiramente, uma instância do modelo Random Forest é criada utilizando a classe RandomForestClassifier e atribuída à variável rfc, representando um modelo de classificação baseado nesta técnica. Em seguida, o modelo é treinado utilizando os dados de treino fornecidos, que são compostos pelos recursos (X_treino) e as classes correspondentes (y_treino). O método fit é chamado na instância do modelo, que ajusta o modelo aos dados de treino, permitindo que ele aprenda os padrões presentes nos dados.

```
fc = RandomForestClassifier()

rfc.fit(X_treino, y_treino)
```

&emsp;&emsp; Após o treinamento, o modelo é utilizado para fazer previsões nos dados de teste, utilizando o método predict do modelo treinado e fornecendo os recursos de teste (X_teste), são geradas as previsões para as classes. Para avaliar o desempenho do modelo, é calculada a acurácia, que é uma métrica que mede a proporção de exemplos corretamente classificados em relação ao total de exemplos. A função accuracy_score é utilizada para calcular a acurácia, recebendo como parâmetros as classes verdadeiras (y_teste) e as classes previstas (y_pred) pelo modelo. A seguir, a função np.unique(y_pred) é utilizada para obter os valores únicos das previsões feitas pelo modelo e, em seguida, a função len() é aplicada para obter o número de valores únicos. Essa linha verifica se o número de classes de saída únicas é igual a 1. Caso a condição seja verdadeira, o modelo tem apenas uma classe de saída possível, onde será exibida uma mensagem na tela usando a função print(). Caso contrário, o código chama a função classification_report(y_teste, y_pred), que recebe os rótulos verdadeiros e as previsões feitas pelo modelo como parâmetros. 

```
y_pred = rfc.predict(X_teste)

acuracia = accuracy_score(y_teste, y_pred)

if len(np.unique(y_pred)) == 1:
    print("O modelo tem apenas uma classe de saída possível.")
	else:
    		classification = classification_report(y_teste, y_pred)
print("\nRelatório de Classificação:")
print(classification)
```

&emsp;&emsp; Abaixo é possível observar o código necessário para criar a matriz de confusão, onde esta é calculada usando a função confusion_matrix(y_teste, y_pred), que recebe como parâmetros os rótulos verdadeiros (y_teste) e as previsões feitas pelo modelo (y_pred). Após calcular a matriz de confusão, uma lista chamada classes é definida, contendo os nomes das classes presentes no problema, que serão utilizados para rotular as classes na matriz de confusão. Em seguida, é criada uma figura de plotagem com as dimensões especificadas usando plt.figure(figsize=(8, 6)). A função sns.heatmap() é chamada para criar o mapa de calor, os parâmetros são: cm é que é a matriz de confusão, annot=True e fmt='g' utilizados para exibir os valores da matriz nas células, cmap='Blues' que define a paleta de cores a ser utilizada no mapa de calor, e por último, os rótulos dos eixos x e y são definidos com base na lista de classes usando os parâmetros xticklabels e yticklabels.

```
cm = confusion_matrix(y_teste, y_pred)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Relatório de Classificação - Sprint 3** <br>
```
             precision    recall  f1-score   support

           0       0.69      0.50      0.58       386
           1       0.76      0.74      0.75       844
           2       0.64      0.77      0.70       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```
&emsp;&emsp; **Matriz de Confusão - Sprint 3** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 40: Matriz de Confusão - Random Forest Sprint 3	
<br>

&emsp;&emsp; **Relatório de Classificação - Sprint 4** <br>
```
             precision    recall  f1-score   support

           0       0.72      0.50      0.59       360
           1       0.65      0.68      0.66       597
           2       0.68      0.76      0.72       651

    accuracy                           0.67      1608
   macro avg       0.68      0.65      0.66      1608
weighted avg       0.68      0.67      0.67      1608
```
&emsp;&emsp; **Matriz de Confusão - Sprint 4** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 41: Matriz de Confusão - Random Forest Sprint 4	
<br>

#### 9.6.3.2 Random Forest - Validação Cruzada

&emsp;&emsp; No próximo código, é realizada uma validação cruzada usando o método cross_val_score, que avalia o desempenho do modelo aplicando-o a múltiplos conjuntos de treinamento e teste. Nesse caso, o modelo Random Forest é usado como o estimador a ser avaliado, onde os dados de entrada bow_model e os rótulos de classe sentimento são passados como parâmetros. A validação cruzada é realizada com 5 folds, ou seja, os dados são divididos em 5 partes iguais e o modelo é treinado e testado em cada combinação dessas partes. As pontuações de validação cruzada são armazenadas na variável scores. O bloco de if já foi explicado anteriormente.

```
rfc = RandomForestClassifier()

scores = cross_val_score(rfc, bow_model, sentimento, cv=5)

print('Pontuações de validação cruzada:', scores)

print('Média da validação cruzada:', scores.mean())

if len(np.unique(y_pred)) == 1:
    print("O modelo tem apenas uma classe de saída possível.")
else:
    classification = classification_report(y_teste, y_pred)
    print("\nRelatório de Classificação:")
    print(classification)
```

&emsp;&emsp;O código da Matriz de Confusão do Random Fores com Cross Validation é o mesmo do apresentado anteriormente. 

&emsp;&emsp; **Relatório de Classificação - Sprint 3** <br>
```
             precision    recall  f1-score   support

           0       0.71      0.50      0.58       386
           1       0.77      0.75      0.76       844
           2       0.63      0.77      0.69       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```
&emsp;&emsp; **Matriz de Confusão - Sprint 3** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_cross_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 42: Matriz de Confusão - Random Forest com validação Sprint 3
<br>

&emsp;&emsp; **Relatório de Classificação - Sprint 4** <br>
```
             precision    recall  f1-score   support

           0       0.74      0.51      0.60       360
           1       0.65      0.70      0.68       597
           2       0.70      0.77      0.73       651

    accuracy                           0.69      1608
   macro avg       0.70      0.66      0.67      1608
weighted avg       0.69      0.69      0.68      1608
```
&emsp;&emsp; **Matriz de Confusão - Sprint 4** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_cross_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 43: Matriz de Confusão - Random Forest com validação Sprint 4
<br>

#### 9.6.3.3 Random Forest - Grid Search

&emsp;&emsp; O código abaixo realiza uma busca em grade com validação cruzada para encontrar os melhores hiperparâmetros para um modelo Random Forest. Inicialmente, é definida uma grade de valores para os hiperparâmetros a serem testados, que são definidos no dicionário parametros. Nesse caso, são considerados três hiperparâmetros: n_estimators (número de estimadores), max_depth (profundidade máxima da árvore) e min_samples_split (número mínimo de amostras para dividir um nó interno). Diferentes valores são fornecidos para cada hiperparâmetro, permitindo que diferentes combinações sejam testadas durante a busca em grade.

```
parametros = {'n_estimators': [100, 200, 300], 
              'max_depth': [None, 10, 20], 
              'min_samples_split': [2, 5, 10]}
```

&emsp;&emsp; 	Em seguida, é criada uma instância do modelo Random Forest utilizando a classe RandomForestClassifier e atribuída à variável "rfc". Após a criação do modelo, é criada uma instância do objeto GridSearchCV, que é responsável por realizar a busca em grade com validação cruzada, este recebe três parâmetros principais: o modelo rfc como estimador, parametros e o número de folds da validação cruzada, definido como 5 através do parâmetro cv=5. Além disso, o parâmetro n_jobs=-1 indica que a busca em grade pode ser executada em paralelo, utilizando todos os núcleos de CPU disponíveis. Por último, o método fit no objeto, usa os dados de entrada bow_model e os rótulos de classe sentimento como parâmetros. O GridSearchCV avalia todas as combinações possíveis dos hiperparâmetros especificados usando a validação cruzada e retorna o melhor modelo encontrado.

```
rfc = RandomForestClassifier()

grid = GridSearchCV(rfc, parametros, cv=5, n_jobs=-1)

grid.fit(bow_model, sentimento)

print('Melhores hiperparâmetros:', grid.best_params_)
print('Melhor pontuação:', grid.best_score_)

if len(np.unique(y_pred)) == 1:
    print("O modelo tem apenas uma classe de saída possível.")
else:
    classification = classification_report(y_teste, y_pred)
    print("\nRelatório de Classificação:")
    print(classification)
```

&emsp;&emsp; O código da Matriz de Confusão do Naive Bayes com Grid Search é o mesmo do apresentado anteriormente. 

&emsp;&emsp; **Relatório de Classificação - Sprint 3** <br>
```
             precision    recall  f1-score   support

           0       0.69      0.50      0.58       386
           1       0.76      0.74      0.75       844
           2       0.64      0.77      0.70       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```
&emsp;&emsp; **Matriz de Confusão - Sprint 3** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_grid_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 44: Matriz de Confusão - Random Forest com Grid Search Sprint 3
<br>

&emsp;&emsp; **Relatório de Classificação - Sprint 4** <br>
```
             precision    recall  f1-score   support

           0       0.74      0.51      0.60       360
           1       0.65      0.70      0.68       597
           2       0.70      0.77      0.73       651

    accuracy                           0.69      1608
   macro avg       0.70      0.66      0.67      1608
weighted avg       0.69      0.69      0.68      1608
```
&emsp;&emsp; **Matriz de Confusão - Sprint 4** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_randomForest_grid_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 45: Matriz de Confusão - Random Forest com Grid Search Sprint 4
<br>

&emsp;&emsp;Todos os modelos apresentados acima foram exportados com a biblioteca pickle, código apresentado abaixo. A primeira linha abre o arquivo, em modo escrita, que será utilizado para armazenar o modelo de rede neural, nesse caso o desenvolvedor pode escolher o nome do arquivo. A segunda linha utiliza a função pickle.dump(), da biblioteca pickle, para salvar o modelo no arquivo aberto anteriormente. A terceira linha abre o arquivo, em modo leitura, para que a função pickle.load() carrega o conteúdo na variável, convertendo os bytes do arquivo novamente em um objeto modelo de rede neural utilizável.
```
with open('nome_escolhido.pkl', 'wb') as arquivo:
    pickle.dump(model, arquivo)
with open('nome_escolhido.pkl', 'rb') as arquivo:
    nome_escolhido = pickle.load(arquivo)
```

### 9.6.4 Conclusão
&emsp;&emsp;Em suma, o Random Forest é um modelo muito promissor que apresentou resultados muito satisfatórios e no final todos os modelos desenvolvidos foram exportados com a biblioteca pickle. 

## 9.7 Rede Neural (Sequência de palavras) - Word2Vec

### 9.7.1 Introdução

&emsp;&emsp; Uma rede neural, também conhecida como rede neural artificial, é um modelo computacional inspirado no funcionamento do cérebro humano. Ela é composta por um conjunto interconectado de unidades de processamento, chamadas de neurônios artificiais ou nós, que trabalham em conjunto para resolver problemas complexos de forma eficiente. Para fins de comparação, foram desenvolvidos códigos que têm a base de dados diferentes. 

### 9.7.2 Método

&emsp;&emsp; O método de sequência de palavras geralmente é utilizado para frases que formam um significado, e a ordem das palavras é crucial para formar o sentido. Então, no caso do projeto, é interessante testar a utilização dessa abordagem.

### 9.7.3 Resultados

&emsp;&emsp; O tópico será dividido da mesma forma que o notebook está dividido para facilitar a compreensão. Além disso, esse modelo foi testado com 3 tipos de bases de dados diferentes (base tratada, word2vec com cbow e word2vec com embedding layer) em dois momentos diferentes (sprint 3 e sprint 4).

#### 9.7.3.1 Leitura da base de dados

&emsp;&emsp; Para a realização com a base tratada, o primeiro processo a ser realizado é a leitura do arquivo csv gerado no notebook do pré processamento, onde o arquivo já está com as etapas realizadas.

```
rede_neural_df = pd.read_csv("caminho do arquivo")
```

&emsp;&emsp; Para a realização dos modelos já vetorizados com o Word2Vec, é necessário somente referenciar a variável usada no notebook.

```
Word2Vec + CBoW: df_vec

Word2Vec + Embedding Layer: df_word2vec
```

#### 9.7.3.1 Separação de treino e teste

&emsp;&emsp; A primeira linha do código é atribuído os valores das colunas "texto_tratado" e "sentimento" do dataframe desejado às variáveis x e y, respectivamente. Essa fase significa que a primeira coluna, "texto_tratado" ou x, contém o texto dos dados a serem classificados e a segunda coluna, "sentimento" ou y, representa a categoria desses dados.

```
x, y = rede_neural_df["texto_tratado"], rede_neural_df["sentimento"]
x, y = df_vec["texto_tratado"], df_vec["sentimento"]
x, y = df_word2vec["texto_tratado"], df_word2vec["sentimento"]
			
obs: somente um desses códigos devem ser utilizados, a depender de qual base será utilizada no desenvolvimento.
```

&emsp;&emsp; As próximas linhas abaixo têm como objetivo realizar um pré-processamento novamente, feito pela rede neural, que contém: LabelEncoder, remoção de algumas palavras e Tokenização.

```
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

words = ["o", "ao", 'aos', 'os', 'a', 'as', 'e', 'um', 'uma','ele', 'ela', 'eles', 'elas', 'do', 'da', 'dos', 'das', 'de', 'no', 'na', 'nos', 'nas', 'pelo', 'pela', 'pelos', 'pelas', 'num', 'numa', 'nuns', 'numas', 'dum', 'duma', 'duns', 'dumas']

x_filter = []

for title in x:
  for word in words:
    title = title.replace(word, '')
  x_filter.append(title)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_filter)
```

&emsp;&emsp; A próxima linha calcula o tamanho do vocabulário, por meio da criação da variável vocab. Além disso, o texto contido na lista x_filter é transformado em uma sequência numérica e logo depois é criado qual será o comprimento máximo dentro da sequência, por meio do max_length. Por último, as sequências de palavras em x_filter são ajustadas para ter o mesmo comprimento máximo através da função pad_sequences de uma biblioteca.

```
vocab = len(tokenizer.word_docs) + 1

x_filter = tokenizer.texts_to_sequences(x_filter)

max_length = max([len(z) for z in x_filter])
x_filter = pad_sequences(x_filter, maxlen=max_length, padding='post')
```

&emsp;&emsp; A última etapa desse tópico é a divisão entre conjuntos de treinamento (x_train e y_train) e teste (x_test e y_test) usando a função train_test_split da biblioteca sklearn.model_selection. Nesse caso foram utilizados 33% dos dados para teste. Além disso, é printado o tamanho dos dados de entrada e saída. 

```
x_train, x_test, y_train, y_test = train_test_split(x_filter, y, test_size=0.33)

print("Tamanho de x:", len(x_filter))
print("Tamanho de y:", len(y))

output - sprint 3: 
Tamanho de x: 9207
Tamanho de y: 9207

output - sprint 4: 
Tamanho de x: 8040
Tamanho de y: 8040
```

#### 9.7.3.3 Criação do modelo

&emsp;&emsp; A função recall implementada acima calcula a métrica de recall para avaliar o desempenho de um modelo de aprendizado de máquina em um problema de classificação binária. A função recebe dois parâmetros: y_true e y_pred, o primeiro representa as verdadeiras classes dos exemplos do conjunto de dados, enquanto o segundo representa as classes previstas pelo modelo. Em seguida, o número de possible_positives é calculado, isso é feito aplicando a função K.clip novamente para limitar os valores de y_true entre 0 e 1, convertendo-os em valores binários. Por último, o recall é calculado dividindo o número de verdadeiros positivos pelo número de positivos possíveis. 

```
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
```

&emsp;&emsp; As linhas abaixo definem como o modelo será estruturado, começando com a criação da variável model que define que o modelo será Sequential() , permitindo o empilhamento de camadas sequencialmente. A segunda linha, adiciona uma camada de embedding, que é responsável por transformar os números inteiros que representam as palavras em vetores densos de números reais. A terceira linha adiciona uma camada de pooling global máxima, que extrai o valor máximo de cada recurso da camada anterior e reduz a dimensão dos dados resultantes para um vetor unidimensional.

```
model = Sequential()
model.add(Embedding(input_dim=vocab, output_dim=80, input_length=max_length, trainable = True))
model.add(GlobalMaxPooling1D())
```

&emsp;&emsp; A próxima linha adiciona uma camada de dropout, que é uma técnica utilizada para prevenir o overfitting. A seguir, é adicionada uma camada densa que possui 3 unidades, correspondendo às 3 classes possíveis de sentimentos. A função de ativação softmax é aplicada para produzir probabilidades de pertencer a cada classe. E, por último, o modelo é compilado e o otimizador adam é usado para ajustar os pesos da rede durante o treinamento. A métrica recall, definida pela função já descrita, é usada para avaliar o desempenho do modelo durante o treinamento.

```
model.add(Dropout(0.3))
model.add(Dense(units = 3, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [recall])
```

&emsp;&emsp; O objeto ModelCheckpoint é responsável por monitorar a acurácia do modelo durante o treinamento e salva apenas os melhores pesos em um arquivo weight.best.hdf5. Por último, o modelo é treinado usando o método fit, os dados de treinamento e de validação são fornecidos. O treinamento é realizado em lotes (batch_size=32) e por um total de 5 épocas. 

```
mc = ModelCheckpoint('weight.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')

model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 32, epochs = 10, callbacks = [mc])
```

##### 9.7.3.3.1 Construção da rede neural com a base tratada - Sprint 3

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com a base tratada da Sprint 3, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
              precision    recall  f1-score   support

           0       0.70      0.68      0.69       662
           1       0.77      0.76      0.77      1358
           2       0.71      0.73      0.72      1019

    accuracy                           0.73      3039
   macro avg       0.72      0.72      0.72      3039
weighted avg       0.73      0.73      0.73      3039
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_base_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 46: Matriz de Confusão - base tratada Sprint 3
<br>

##### 9.7.3.3.2 Construção da rede neural com Word2Vec + CBoW - Sprint 3

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com o Word2Vec + CBoW da Sprint 3, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
             precision    recall  f1-score   support

           0       0.70      0.70      0.70       633
           1       0.79      0.74      0.77      1308
           2       0.71      0.77      0.74      1098

    accuracy                           0.74      3039
   macro avg       0.73      0.74      0.73      3039
weighted avg       0.74      0.74      0.74      3039
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_word2vec_cbow_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 47: Matriz de Confusão - Word2Vec + CBoW Sprint 3
<br>

##### 9.7.3.3.3 Construção da rede neural com Word2Vec + Embedding Layer - Sprint 3

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com o Word2Vec + Embedding Layer da Sprint 3, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
             precision    recall  f1-score   support

           0       0.67      0.69      0.68       632
           1       0.76      0.75      0.76      1321
           2       0.72      0.72      0.72      1086

    accuracy                           0.73      3039
   macro avg       0.72      0.72      0.72      3039
weighted avg       0.73      0.73      0.73      3039
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_word_embedding_sprint3.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 48: Matriz de Confusão - Word2Vec + Embedding Layer Sprint 3
<br>

##### 9.7.3.3.4 Construção da rede neural com a base tratada - Sprint 4

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com a base tratada da Sprint 4, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
             precision    recall  f1-score   support

           0       0.67      0.66      0.67       624
           1       0.72      0.64      0.68       990
           2       0.69      0.78      0.73      1040

    accuracy                           0.70      2654
   macro avg       0.70      0.69      0.69      2654
weighted avg       0.70      0.70      0.70      2654
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_base_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 49: Matriz de Confusão - Base tratada Sprint 4
<br>

##### 9.7.3.3.5 Construção da rede neural com Word2Vec + CBoW - Sprint 4

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com o Word2Vec + CBoW da Sprint 4, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
             precision    recall  f1-score   support

           0       0.66      0.63      0.65       655
           1       0.71      0.64      0.68       969
           2       0.69      0.77      0.73      1030

    accuracy                           0.69      2654
   macro avg       0.69      0.68      0.68      2654
weighted avg       0.69      0.69      0.69      2654
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_word_cbow_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 50: Matriz de Confusão - Word2Vec + CBoW Sprint 4
<br>

##### 9.7.3.3.6 Construção da rede neural com Word2Vec + Embedding Layer - Sprint 4

&emsp;&emsp; Abaixo é possível observar o relatório de classificação do modelo com o Word2Vec + Embedding Layer da Sprint 4, onde o 0 é negativo, o 1 é neutro e o 2 é positivo. Após isso, é gerada uma matriz de confusão. 

&emsp;&emsp; **Relatório de Classificação** <br>
```
             precision    recall  f1-score   support

           0       0.68      0.64      0.66       676
           1       0.68      0.69      0.69       980
           2       0.70      0.72      0.71       998

    accuracy                           0.69      2654
   macro avg       0.69      0.68      0.69      2654
weighted avg       0.69      0.69      0.69      2654
```
&emsp;&emsp; **Código da matriz de confusão:** <br>
```
cm = confusion_matrix(y_test, y_pred_classes)
classes = ['Classe 1', 'Classe 2', 'Classe 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot
=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()
```

&emsp;&emsp; **Matriz de Confusão** <br>
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/matriz_confusao_redeNeuralSeq_word_embedding_sprint4.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 51: Matriz de Confusão - Word2Vec + Embedding Layer Sprint 4
<br>

#### 9.7.3.4 Exportação com a biblioteca pickle

&emsp;&emsp; A primeira linha abre o arquivo, em modo escrita, que será utilizado para armazenar o modelo de rede neural, nesse caso o desenvolvedor pode escolher o nome do arquivo. A segunda linha utiliza a função pickle.dump(), da biblioteca pickle, para salvar o modelo no arquivo aberto anteriormente. A terceira linha abre o arquivo, em modo leitura, para que a função pickle.load() carrega o conteúdo na variável, convertendo os bytes do arquivo novamente em um objeto modelo de rede neural utilizável. Todos os modelos foram salvos por meio do código abaixo.

```
with open('nome_escolhido.pkl', 'wb') as arquivo:
    pickle.dump(model, arquivo)
with open('nome_escolhido.pkl', 'rb') as arquivo:
    'nome_escolhido = pickle.load(arquivo)
```

### 9.7.4 Exportação com a biblioteca pickle

&emsp;&emsp;Os códigos apresentados mostram o processo de construção e treinamento de uma rede neural para classificação de texto, além de fornecer uma maneira de salvar e carregar o modelo treinado para uso posterior, com a utilização da biblioteca pickle.
































