# **6. Criação do Modelo - Bag of Words**

## 6.1 Introdução

### 6.1.1 Análise descritiva
TEXTO AQUI 
### 6.1.2 Pré - processamento
TEXTO AQUI 
### 6.1.3 Modelo Bag of words
TEXTO AQUI 

## 6.2 Método

### 6.2.1 Análise descritiva
TEXTO AQUI 
### 6.2.2 Pré - processamento
TEXTO AQUI 
### 6.2.3 Modelo Bag of words
TEXTO AQUI 

## 6.3 Resultados

### 6.3.1 Análise descritiva

Na análise descritiva dos dados, foram explorados 2 tipos de gráficos: gráficos de pizza e barras. Utilizando técnicas de visualização, foi possível apresentar informações relevantes e obter insights sobre os dados em questão. Abaixo serão descritos os gráficos e os resultados obtidos. 

##### 6.3.1.1 Autores
A seguir é mostrado o código utilizado para plotar o primeiro gráfico de barras, com o objetivo de demonstrar quais são os usuários (autores) que mais comentam nos posts do BTG Pactual. 
<br>
```
    autor_counts = data_limpo['autor'].explode().value_counts()
    plt.figure(figsize=(10, 6))
    autor_counts.head(20).plot(kind='bar')
    plt.xlabel('Autores')
    plt.ylabel('Contagem')
    plt.title('Autores que mais comentam')
    plt.show()
```
Na primeira linha é criada uma nova variável chamada autor_counts. Ela utiliza a coluna "autor" do dataframe data_limpo. A função explode() é aplicada para transformar uma coluna de listas em várias linhas, e em seguida, a função value_counts() é aplicada para contar a ocorrência de comentários daqueles usuários. A segunda linha especifica um tamanho para a figura.
<br>
Na terceira linha é utilizado o head(20) para selecionar somente os 20 primeiros valores da variável autor_counts. Em seguida, o método plot é chamado com o parâmetro kind='bar', indicando que um gráfico de barras deve ser criado. 
<br>
Na quarta e quinta linha é criado um rótulo no eixo x e y do gráfico com o texto "Autores" e “Contagem”, respectivamente. Na sexta linha é definido qual é o título do gráfico: “Autores que mais comentam”. Por último, a sétima linha exibe o gráfico abaixo.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/autores.png"> <br>

Com esse gráfico foi possível observar que, na maioria das vezes, tem um padrão muito claro de frequência de comentários, o que significa que a empresa mantém um público específico que também é muito engajado. Apesar disso, foi criada a hipótese de que, pelo fato do primeiro usuário (@amgcapitalinvest) ser uma empresa credenciada pelo BTG, ela marca o banco nos seus posts, referenciando os créditos, é possível interpretar que talvez não sejam somente comentários. 
<br>

##### 6.3.1.2 Palavras mais frequentes
A seguir é mostrado o código utilizado para plotar o segundo gráfico de barras, com o objetivo de demonstrar quais são as palavras mais frequentes utilizadas nos comentários.
<br>
```
  word_counts = data_limpo['texto_tratado'].explode().value_counts()
  plt.figure(figsize=(10, 6))
  word_counts.head(20).plot(kind='bar')
  plt.xlabel('Palavra')
  plt.ylabel('Contagem')
  plt.title('Top 20 Palavras Mais Frequentes')
  plt.show()
```
O código se inicia criando uma nova variável chamada word_counts. Ela utiliza a coluna "texto_tratado" do dataframe data_limpo. Em seguida, a função explode() é aplicada para transformar uma coluna de listas em várias linhas, onde cada valor da lista é tratado como uma nova observação. Por fim, a função value_counts() é utilizada para contar a ocorrência de cada palavra e gerar a contagem de palavras.
<br>
A seguir, a segunda linha cria uma nova figura de plotagem com o tamanho específico, para que se torne melhor a aparência visual. A terceira linha seleciona os primeiros 20 valores da variável word_counts utilizando o método head(20). Além disso, é utilizado o método plot com o parâmetro kind='bar', indicando o tipo do gráfico a ser gerado, nesse caso o de barras.
<br>
A quarta, quinta e sexta linha são utilizadas para definir as legendas, eixo x e y (quarta e quinta linha) e o título (sexta linha). Por último, o método plt.show() exibe o gráfico abaixo. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/palavras.png"> <br>

Com esse gráfico é possível observar que entre as 20 palavras, 7 delas estão diretamente relacionadas ao banco: “btgpactual”, “invest”, “btg”, “banc”, “merc”, “financeir”, “pactual”. Isso pode significar que geralmente as pessoas estão respondendo o post com os assuntos neles descritos, que, na maioria das vezes, tem como tema o mercado financeiro. Além disso, foi criada uma hipótese que a palavra “btgpactual” se diz respeito à marcação da conta do banco e não necessariamente falando sobre ele, já que as palavras: “btg” e “pactual” estão entre as 20 palavras mais frequentes. 
<br>

##### 6.3.1.3 Tipos de interação
A seguir é mostrado o código utilizado para plotar o terceiro gráfico, que com o objetivo de demonstrar a diferença entre os tipos de interação presentes no dataframe.
<br>
```
  count_interactions = data_limpo['tipoInteracao'].value_counts()
  plt.figure(figsize=(8, 6))
  count_interactions.plot(kind='pie', autopct='%1.1f%%')
  plt.title('Tipos de Interação')
  plt.ylabel('')
  plt.show()
```
Na primeira linha, é criada uma nova variável chamada count_interactions, que tem como função utilizar a coluna "tipoInteracao" do dataframe data_limpo. E com isso, a função value_counts() é aplicada para contar a ocorrência de cada tipo de interação. A seguir é especificado qual é o tamanho do gráfico.
<br>
Na terceira linha, a variável count_interactions usa o método plot com o parâmetro kind='pie', que indica o tipo de gráfico que deve ser gerado, nesse caso de pizza. Além disso, o parâmetro autopct='%1.1f%%' é utilizado para exibir a porcentagem de cada fatia no gráfico.
<br>
Na quarta linha é definido um título para o gráfico, com o texto "Tipos de Interação". E a seguir, na quarta linha, o rótulo do eixo y é removido, por ser um gráfico de pizza e as porcentagens já estão sendo mostradas. Por último, o gráfico abaixo é exibido na saída.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/interacao.png"> <br>

O gráfico acima demonstra que, caso a hipótese de ter repost dos posts do BTG esteja certa, o dataframe está, em sua maioria com esses casos, o que torna preocupante, já que a ideia é que o projeto analise comentários dos posts. Além disso, pode-se observar uma diferença significativa entre “comentários” e “resposta”. 

##### 6.3.1.4 Classificação de sentimento
Para esse tipo de gráfico foram criados 2 gráficos, para isso será demonstrado os 2 códigos e a diferença entre eles. 
```
  count_sentimentos = data_limpo['sentimento'].value_counts()
  # Gráfico de pizza
  plt.figure(figsize=(8, 6))
  count_sentimentos.plot(kind='pie', autopct='%1.1f%%')
  plt.title('Tipos de Sentimento')
  plt.ylabel('')
  plt.show()
  # Gráfico de barras
  plt.figure(figsize=(10, 6))
  count_sentimentos.plot(kind='bar')
  plt.xlabel('Sentimentos')
  plt.ylabel('Contagem')
  plt.title('Tipos de Sentimento')
  plt.show()
```
A primeira linha é criada uma nova variável chamada count_sentimentos, que utiliza a coluna "sentimento" do dataframe data_limpo. A função value_counts() é aplicada para contar a ocorrência de cada tipo de sentimento. Essa linha pertence aos 2 tipos de gráficos, isso porque ela só está definindo a variável e função que serão utilizadas posteriormente.  
<br>
O primeiro gráfico gerado é o de pizza:
<br>
A primeira linha específica do gráfico de pizza define qual será o tamanho da figura. A segunda linha chama a variável count_sentimentos, utilizando o método plot com o parâmetro kind='pie', indicando o tipo de gráfico, além disso, o parâmetro autopct='%1.1f%%' é utilizado para exibir a porcentagem de cada fatia no gráfico. A seguir, é definido o título do gráfico e remove o rótulo do eixo y, já que não será utilizado. Por último, o método show exibe o gráfico a seguir na saída.  
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_pizza.png"> <br>
O segundo gráfico gerado é o de barras:
<br>
A primeira linha do gráfico de barra define qual será o tamanho da figura que será gerada no final do código. A seguir, a variável count_sentimentos é plotada utilizando o método plot com o parâmetro kind='bar', indicando o tipo de gráfico, essa linha que diferencia os tipos de gráficos. As próximas 3 linhas são usadas para definir os rótulos dos eixo x e y e o título do gráfico. A última linha exibe o gráfico a seguir na saída. 
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_barra.png"> <br>
Analisando os gráficos é possível observar que a quantidade de comentários neutros é maior que os outros dois, pode-se interpretar que essa métrica é ruim para os dados, e com isso podemos chegar em duas hipóteses: 1. 43% dos comentários não causam nenhum tipo de sentimento para as pessoas; ou 2. A classificação feita está equivocada, caso os posts causem algum tipo de sentimento. Além disso, a quantidade de comentários positivos é quase o dobro do negativo, o que se pode referir que os usuários estão se sentindo contentes com os serviços prestados. 

### 6.3.2 Pré - processamento
Abaixo serão descritos cada etapa do pré - processamento.

##### 6.3.2.1 Tratamento dos dados

Um dos primeiros tratamentos de dados que foi utilizado, foi o tratamento que retira as aspas duplas (“”) dos nomes das colunas da base de dados, já que anteriormente as colunas estavam da seguinte forma: “texto”, após esse tratamento, ficou apenas texto, como demonstra o código abaixo:
```
data = data.rename(columns={'"anomalia"' : 'anomalia', '"dataPublicada"' : 'dataPublicada', '"autor"' : 'autor', '"texto"' : 'texto', '"sentimento"' : 'sentimento', '"tipoInteracao"' : 'tipoInteracao', '"probabilidadeAnomalia"' : 'probabilidadeAnomalia', '"linkPost"' : 'linkPost', '"processado"' : 'processado',  '"contemHyperlink"' : 'contemHyperlink' })
```
Esse tratamento facilita o trabalho de chamar os textos das colunas de uma maneira mais simples, sem a necessidade de ter que colocar aspas, podendo chamar o texto diretamente.
<br>
O segundo tratamento realizado utilizou a função data.describe(), e com isso foi possível identificar que a coluna “processado” não agrega valor, uma vez que todos os seus valores são iguais a zero. Portanto, essa coluna foi removida da base de dados. Abaixo é possível ver o output desta função. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/data_describe.jpg"> <br>

Com isso, foi possível descartar essa coluna utilizando a função data.drop(). Vale ressaltar que as colunas “id” e “dataPublicada” também foram removidas da base de dados, uma vez que não possuem tanta relevância para uma análise de sentimento que tem como principal embasamento os textos, como mostra o código abaixo.
```
data_dropado = data.drop(['processado', 'id', 'dataPublicada'], axis=1)
data_dropado.head(3)
```
O terceiro tratamento realizado foi a remoção do autor @btgpactual, da coluna “autor”, foi possível removê-lo através de uma função que remove apenas o autor mencionado.
```
data_limpo = data_dropado.loc[data_dropado['autor'] != 'btgpactual']
data_limpo
```
Esse tratamento é necessário para o projeto, pois os comentários vindos desse autor não são tão relevantes para análise de sentimentos, uma vez que a maioria são respostas a comentários ou legendas dos posts.

##### 6.3.2.2 Tokenização

Para começar o pré - processamento pensando no modelo de análise de sentimento, é necessário separar as palavras dos textos em tokens, e o código abaixo define a função necessária para realizar esse processo.
```
def tokenizer(comment):
    if isinstance(comment, str):
        tokens = nltk.word_tokenize(comment)
        return tokens
    else:
        return []
```
A função acima realiza o processo descrito referenciando a biblioteca nltk.word_tokenize.

##### 6.3.2.3 Remoção de stopwords

Já que as palavras que são consideradas como stopwords não tem uma importância para o sentido do texto e elas ocupam a maior parte dos tokens, essa etapa foi realizada por meio do código abaixo:
```
def remove_stopwords(tokens):
    if isinstance(tokens, list): 
        comments_filtered = []
        for token in tokens:
          tk = token.lower()
          if tk not in stopwords:
              comments_filtered.append(tk)
        return comments_filtered
    else:
        return []
```
A função acima referencia a biblioteca para que as palavras classificadas sejam removidas do conjunto de tokens. 

##### 6.3.2.4 Remoção de acentos

Da mesma forma que algumas palavras não têm importância para a análise, a pontuação e caracteres especiais também não tem, por isso a função abaixo retira esses caracteres. 
```
def remover_pontuacao(tokens):
    tokens_sem_pontuacao = []
    for token in tokens:
        token_sem_pontuacao = re.sub(r'[^\w\s]', '', token)
        if token_sem_pontuacao != '':
            tokens_sem_pontuacao.append(token_sem_pontuacao)
    return tokens_sem_pontuacao
```

##### 6.3.2.5 Pipeline

No pipeline foi dividido cada uma das funções em células separadas e depois é executado todas na ordem correta. Essa etapa permite que as funções sejam executadas na ordem correta, garantindo a consistência e a precisão dos resultados, e caso a ordem precise mudar, é mais simples fazer a alteração, essa organização torna o processo mais simples de entender e escalável.
<br>
Na parte de definição de funções, foi definida as funções que serão usadas no pipeline. As funções em questão são: tokenizer(); remove_stopwords(); remover_pontuacao(); stemming(), por fim, a função pipeline() executa cada uma das funções em ordem. Como as funções já foram apresentadas anteriormente, a seguir será mostrada a função pipeline():
```
def pipeline(comment):
      tokens = tokenizer(comment)
      tokens_filtered = remove_stopwords(tokens)
      tokens_no_punct = remover_pontuacao(tokens_filtered)
      stemmed_tokens = stemming(tokens_no_punct)
      return stemmed_tokens
```
Por fim, foram realizados alguns testes de função para garantir que o fluxo do pipeline estava operando adequadamente, para isso, foi criado um novo dataframe com uma coluna chamada “texto_tratado”, na qual está o resultado de todos os textos após passar pela função pipeline(). 
```
data_limpo['texto_tratado'] = data_limpo['texto'].apply(pipeline)
```
Em seguida, a função é executada com a frase "Estamos fazendo um projeto pro BTG!", com a intenção de encontrar potenciais falhas no algoritmo, o código abaixo demonstra esse processo.
```
comment = "Estamos fazendo um projeto pro BTG!"
preprocessed_comment = pipeline(comment)
print(preprocessed_comment)

output: ['faz', 'projet', 'pro', 'btg']
```
Com isso, é possível notar que a função remove a palavra “Estamos” na frase e que o algoritmo não removeu a palavra “pro”, o que significa que abreviações de palavras e gírias podem prejudicar a acurácia do algoritmo.
A imagem abaixo exemplifica todos os processos descritos acima e conta com exemplos de inputs e outputs.

COLOCAR IMAGEM - PIPELINE

### 6.3.3 Modelo Bag of words

COLOCAR IMAGEM - PIPELINE

Após o corpus  dos textos terem passado pelo pipeline, chega o momento de analisar as repetições de acordo com cada comentário feito, por meio da técnica Bag of Words (BoW) utilizada em processamento de linguagem natural (PLN). Essa técnica é utilizada para representar um texto como um conjunto de palavras desordenadas, ignorando a ordem e a estrutura gramatical das frases.  Nesse modelo, cada palavra única do texto é transformada em uma "feature" (característica), e a frequência de cada palavra no texto é usada como um valor numérico para a feature correspondente.
<br>
Por exemplo, a frase "O gato preto pulou o muro" seria representada como um conjunto de palavras desordenadas: `'o', 'gato', 'preto', 'pulou', 'o', 'muro'`. A frequência de cada palavra é contada, e o resultado é um vetor numérico que representa a frequência de cada palavra na frase. O modelo Bag of Words é uma técnica simples e eficiente para representar textos em formato vetorial, o que permite utilizá-los em algoritmos de aprendizado de máquina. 
<br>
Assim, abaixo é possível visualizar o código necessário para realizar essa vetorização e o output dele:
```
def bow(comentarios): 
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    bow_model = vectorizer.fit_transform(comentarios)
    bow_df = pd.DataFrame(bow_model.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df
```
COLOCAR IMAGEM - OUTPUT

Abaixo é demonstrado um exemplo resultante desta tabela, a qual possui um total de 12.193 linhas, que estão de acordo com cada comentário do csv disponibilizado pelo cliente, além de 24.331 colunas, que foram as palavras chaves selecionadas.
```
df['conf'].value_counts() 
0    11795
1      396
2        2
Name: conf, dtype: int64
```
Neste exemplo, é possível perceber que o termo ‘conf’  se repete uma vez, em 396 comentários diferentes, e se repete duas vezes em 2 comentários diferentes. Dessa forma, percebe-se como a função consegue selecionar palavras chaves que estão contidas nas diversas frases do dataframe.



## 6.4 Conclusão

### 6.4.1 Análise descritiva
Esta análise descritiva dos gráficos proporciona uma compreensão mais profunda dos dados, permitindo identificar insights e tomar decisões. É importante ressaltar que as conclusões obtidas são interpretadas considerando o contexto específico dos dados e as questões de pesquisa em análise.
### 6.4.2 Pré - processamento
O pré-processamento dos dados é fundamental para garantir a qualidade e a confiabilidade das análises posteriores, contribuindo para um melhor entendimento dos dados e para a obtenção de resultados mais precisos e significativos.
### 6.4.3 Modelo Bag of words
Com a aplicação do Modelo Bag of Words (BoW) é possível perceber a capacidade de seleção de palavras para a futura implementação na Machine Learning desenvolvida. O objetivo do projeto é demonstrado a partir da imagem abaixo:

COLOCAR IMAGEM - PIPELINE

Porém, foi possível analisar que é necessário uma renovação no tratamento dos dados e exclusão de determinadas palavras, já que foi percebido que havia uma alta diversidade de termos que estão exclusos e/ou outros que permanecerão nas frases e não deveriam permanecer. Abaixo há exemplo desta análise:

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

Além disso, foi feita uma plotagem de uma nuvem de palavras para ser mais intuitiva a visualização dos termos que serão necessários passar por um tratamento.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/nuvem_palavras.png"> <br>

Assim, o próximo passo é um retratamento dos textos para ter melhor desenvolvimento e aplicação no momento de construção da Inteligência Artificial.
