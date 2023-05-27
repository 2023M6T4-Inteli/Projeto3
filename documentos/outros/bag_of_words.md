# **6. Criação do Modelo - Bag of Words**

## 6.1 Introdução

### 6.1.1 Análise descritiva
&emsp;&emsp; A análise descritiva é uma técnica estatística que pode ser aplicada em diferentes áreas, incluindo a análise de sentimentos. No contexto do projeto proposto pelo BTG de análise de sentimentos realizado a partir de comentários de usuários em publicações no _Instagram_ do banco, a análise descritiva é utilizada para descrever e resumir as principais características dos dados coletados. <br>
&emsp;&emsp; Por meio desta, é possível obter informações sobre: 
- O número total de comentários coletados; </br>
- A distribuição de sentimentos positivos, negativos e neutros expressos pelos usuários; </br>
- As palavras mais frequentes nos comentários; </br>
- Os usuários que mais realizaram comentários; </br>

&emsp;&emsp; Essas informações são cruciais para compreender melhor a percepção dos usuários em relação ao banco e para orientar futuras estratégias de comunicação e relacionamento com o público, garantindo uma maior assertividade em futuras publicações do banco BTG.

### 6.1.2 Pré - processamento
&emsp;&emsp; O pré-processamento de dados no contexto do PLN refere-se a uma série de etapas de preparação que os dados textuais devem passar antes de serem usados em um modelo de aprendizado de máquina. Essas etapas visam limpar, organizar e estruturar os dados textuais para que sejam mais facilmente compreendidos pelo modelo. Algumas etapas importantes do pré- processamento são: 
- Tokenização; </br> 
- Tratamento de abreviações; </br> 
- Tratamento de emoji; </br> 
- Remoção de stopwords; </br>
- Remoção de alfanuméricos; </br>
- Lematização; </br> 

&emsp;&emsp; Além disso, foi realizado um tratamento dos dados e a definição de uma função _pipeline_.
 
### 6.1.3 Modelo Bag of words
&emsp;&emsp; O modelo _Bag of Words_ é uma das várias ferramentas de vetorização de frases e palavras, processo que é de suma importância para o desenvolvimento de um modelo PLN (Processamento de Linguagem Natural), visto que o modelo de _machine learning_ só pode receber números como _inputs_. 

## 6.2 Método

### 6.2.1 Análise descritiva
&emsp;&emsp; Os comentários realizados pelos usuários nas publicações do banco BTG são uma fonte valiosa de informações para entender como os clientes se sentem em relação aos serviços oferecidos pela instituição financeira. Para realizar a análise desses dados, foram utilizados diversos métodos de tratamento de dados, que serão descritos no tópico 6.2.2 Pré - processamento. </br>
&emsp;&emsp; Para visualizar as informações de maneira clara e acessível, foram utilizados gráficos de barra e pizza. Os gráficos de barra foram utilizados para a visualização das palavras mais frequentes encontradas nos comentários; identificação dos autores mais ativos e o tipo de sentimento causado (positivo, neutro ou negativo). Já nos gráficos de pizza destacam os tipos de interação mais utilizados, alternando entre comentários, menções e respostas e também tipos de sentimentos expressos pelos usuários. </br>
&emsp;&emsp; Para realizar a análise e a visualização desses dados, foram utilizadas bibliotecas como: _Matplotlib_, que é uma biblioteca de visualização de dados em _Python_, além de bibliotecas notáveis como é o caso do _pandas_, _numpy_ e a _nltk_. Com essas ferramentas, foi possível obter _insights_ valiosos sobre a percepção dos clientes em relação ao Banco BTG e identificar áreas que precisam de melhorias.

### 6.2.2 Pré - processamento
&emsp;&emsp; A etapa de pré - processamento, como dito anteriormente, foi dividida em 6 etapas:
- **Tratamento de dados**: processo que envolve a manipulação, limpeza, enriquecimento e transformação de dados de forma a torná-los mais úteis e adequados para a análise. Onde foram realizados os processos de: Mudança dos nomes das colunas - retirada das aspas (“”); Retirada de algumas colunas que se mostraram não necessárias para o projeto; Remoção dos comentários do banco (autor: @btgpactual);
- **Tokenização**: processo de dividir um texto em unidades menores chamadas _tokens_. Esses _tokens_ podem ser palavras individuais ou partes menores de palavras, como prefixos ou sufixos;
- **Tratamento de abreviações**: processo de transformar palavras abreviadas, muito utilizada em redes sociais, para sua expansão;
- **Remoção de pontuações**: processo de retirar os caracteres de pontuação para reduzir o tamanho do vocabulário e evitar ruídos;
- **Tratamento de emoji**: processo de transformar símbolos de emojis para o seu significado; 
- **Remoção de stopwords**: palavras que podem ser consideradas irrelevantes para o conjunto de resultados a ser exibido;
- **Remoção de alfanuméricos**: processo de retirar os caracteres de pontuação para reduzir o tamanho do vocabulário e evitar ruídos;
- **Lematização**: processo de normalização das palavras, reduzindo a variabilidade e simplificando a análise e compreensão do texto;
- **Pipeline**: sequência de etapas ou processos interligados que são aplicados aos dados durante o fluxo de trabalho. </br>
&emsp;&emsp; Para essa etapa foi utilizada a biblioteca do _NLTK_, para a tokenização e remoção de stopwords, a biblioteca _emoji_ para o tratamento do mesmo, o _spacy_, para a lematização e o _pandas_ para a leitura e tratamento dos dados. 
		
### 6.2.3 Modelo Bag of words
&emsp;&emsp; Como última etapa de manipulação de dados antes do uso do modelo de _Machine Learning_ para a classificação de resultados temos a vetorização dos comentários, processo que nessa _pipeline_ foi conduzido pelo modelo _Bag of Words (BoW)_. O modelo BoW consiste na elaboração de uma matriz a partir de um vocabulário de todos os vocábulos presentes nos textos, enquanto que cada linha será um comentário que se deseja vetorizar. É importante notar que esse modelo é menos robusto, considerando apenas a frequência de palavras em cada frase e não os sentidos semânticos. <br>
&emsp;&emsp; Para essa etapa, foi utilizada uma instância da classe `CountVectorizer()`, e seus métodos, da biblioteca _sklearn_ (scikit-learn) a fim de que fosse gerado um vocabulário e as respectivas correspondências para cada comentário.

## 6.3 Resultados

### 6.3.1 Análise descritiva
&emsp;&emsp; Na análise descritiva dos dados, foram explorados 2 tipos de gráficos: gráficos de pizza e barras. Utilizando técnicas de visualização, foi possível apresentar informações relevantes e obter _insights_ sobre os dados em questão. Abaixo serão descritos os gráficos e os resultados obtidos. 

##### 6.3.1.1 Autores
&emsp;&emsp; A seguir é mostrado o código utilizado para plotar o primeiro gráfico de barras, com o objetivo de demonstrar quais são os usuários (autores) que mais comentam nos posts do BTG Pactual. 
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
&emsp;&emsp; Na primeira linha é criada uma nova variável chamada `autor_counts`. Ela utiliza a coluna "autor" do dataframe `data_limpo`. A função `explode()` é aplicada para transformar uma coluna de listas em várias linhas, e em seguida, a função `value_counts()` é aplicada para contar a ocorrência de comentários daqueles usuários. A segunda linha especifica um tamanho para a figura.
<br>
&emsp;&emsp; Na terceira linha é utilizado o `head(20)` para selecionar somente os 20 primeiros valores da variável `autor_counts`. Em seguida, o método `plot` é chamado com o parâmetro `kind='bar'`, indicando que um gráfico de barras deve ser criado. 
<br>
&emsp;&emsp; Na quarta e quinta linha é criado um rótulo no eixo x e y do gráfico com o texto "Autores" e “Contagem”, respectivamente. Na sexta linha é definido qual é o título do gráfico: “Autores que mais comentam”. Por último, a sétima linha exibe o gráfico abaixo.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/autores.png"> <br>

&emsp;&emsp; Com esse gráfico foi possível observar que, na maioria das vezes, tem um padrão muito claro de frequência de comentários, o que significa que a empresa mantém um público específico que também é muito engajado. Apesar disso, foi criada a hipótese de que, pelo fato do primeiro usuário (@amgcapitalinvest) ser uma empresa credenciada pelo BTG, ela marca o banco nos seus posts, referenciando os créditos, é possível interpretar que talvez não sejam somente comentários. 
<br>

##### 6.3.1.2 Palavras mais frequentes
&emsp;&emsp; A seguir é mostrado o código utilizado para plotar o segundo gráfico de barras, com o objetivo de demonstrar quais são as palavras mais frequentes utilizadas nos comentários.
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
&emsp;&emsp; O código se inicia criando uma nova variável chamada `word_counts`. Ela utiliza a coluna "texto_tratado" do dataframe `data_limpo`. Em seguida, a `função explode()` é aplicada para transformar uma coluna de listas em várias linhas, onde cada valor da lista é tratado como uma nova observação. Por fim, a função `value_counts()` é utilizada para contar a ocorrência de cada palavra e gerar a contagem de palavras.
<br>
&emsp;&emsp; A seguir, a segunda linha cria uma nova figura de plotagem com o tamanho específico, para que se torne melhor a aparência visual. A terceira linha seleciona os primeiros 20 valores da variável `word_counts` utilizando o método `head(20)`. Além disso, é utilizado o método `plot` com o parâmetro `kind='bar'`, indicando o tipo do gráfico a ser gerado, nesse caso o de barras.
<br>
&emsp;&emsp; A quarta, quinta e sexta linha são utilizadas para definir as legendas, eixo x e y (quarta e quinta linha) e o título (sexta linha). Por último, o método `plt.show()` exibe o gráfico abaixo. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/palavras.png"> <br>

&emsp;&emsp; Com esse gráfico é possível observar que entre as 20 palavras, 7 delas estão diretamente relacionadas ao banco: “btgpactual”, “invest”, “btg”, “banc”, “merc”, “financeir”, “pactual”. Isso pode significar que geralmente as pessoas estão respondendo o post com os assuntos neles descritos, que, na maioria das vezes, tem como tema o mercado financeiro. Além disso, foi criada uma hipótese que a palavra “btgpactual” se diz respeito à marcação da conta do banco e não necessariamente falando sobre ele, já que as palavras: “btg” e “pactual” estão entre as 20 palavras mais frequentes. 
<br>

##### 6.3.1.3 Tipos de interação
&emsp;&emsp; A seguir é mostrado o código utilizado para plotar o terceiro gráfico, que com o objetivo de demonstrar a diferença entre os tipos de interação presentes no dataframe.
<br>
```
  count_interactions = data_limpo['tipoInteracao'].value_counts()
  plt.figure(figsize=(8, 6))
  count_interactions.plot(kind='pie', autopct='%1.1f%%')
  plt.title('Tipos de Interação')
  plt.ylabel('')
  plt.show()
```
&emsp;&emsp; Na primeira linha, é criada uma nova variável chamada `count_interactions`, que tem como função utilizar a coluna "tipoInteracao" do dataframe `data_limpo`. E com isso, a função `value_counts()` é aplicada para contar a ocorrência de cada tipo de interação. A seguir é especificado qual é o tamanho do gráfico.
<br>
&emsp;&emsp; Na terceira linha, a variável `count_interactions` usa o método `plot` com o parâmetro `kind='pie'`, que indica o tipo de gráfico que deve ser gerado, nesse caso de pizza. Além disso, o parâmetro `autopct='%1.1f%%'` é utilizado para exibir a porcentagem de cada fatia no gráfico.
<br>
&emsp;&emsp; Na quarta linha é definido um título para o gráfico, com o texto "Tipos de Interação". E a seguir, na quarta linha, o rótulo do eixo y é removido, por ser um gráfico de pizza e as porcentagens já estão sendo mostradas. Por último, o gráfico abaixo é exibido na saída.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/interacao.png"> <br>

&emsp;&emsp; O gráfico acima demonstra que, caso a hipótese de ter repost dos posts do BTG esteja certa, o dataframe está, em sua maioria com esses casos, o que torna preocupante, já que a ideia é que o projeto analise comentários dos posts. Além disso, pode-se observar uma diferença significativa entre “comentários” e “resposta”. 

##### 6.3.1.4 Classificação de sentimento
&emsp;&emsp; Para esse tipo de gráfico foram criados 2 gráficos, para isso será demonstrado os 2 códigos e a diferença entre eles. 
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
&emsp;&emsp; A primeira linha é criada uma nova variável chamada `count_sentimentos`, que utiliza a coluna "sentimento" do dataframe `data_limpo`. A função `value_counts()` é aplicada para contar a ocorrência de cada tipo de sentimento. Essa linha pertence aos 2 tipos de gráficos, isso porque ela só está definindo a variável e função que serão utilizadas posteriormente.  
<br>
**O primeiro gráfico gerado é o de pizza**:
<br>
&emsp;&emsp; A primeira linha específica do gráfico de pizza define qual será o tamanho da figura. A segunda linha chama a variável `count_sentimentos`, utilizando o método `plot` com o `parâmetro kind='pie'`, indicando o tipo de gráfico, além disso, o parâmetro `autopct='%1.1f%%'` é utilizado para exibir a porcentagem de cada fatia no gráfico. A seguir, é definido o título do gráfico e remove o rótulo do eixo y, já que não será utilizado. Por último, o método `show` exibe o gráfico a seguir na saída.  
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_pizza.png"> <br>
**O segundo gráfico gerado é o de barras**:
<br>
&emsp;&emsp; A primeira linha do gráfico de barra define qual será o tamanho da figura que será gerada no final do código. A seguir, a variável `count_sentimentos` é plotada utilizando o método `plot` com o parâmetro `kind='bar'`, indicando o tipo de gráfico, essa linha que diferencia os tipos de gráficos. As próximas 3 linhas são usadas para definir os rótulos dos eixo x e y e o título do gráfico. A última linha exibe o gráfico a seguir na saída. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_coluna.png"> <br>
&emsp;&emsp; Analisando os gráficos é possível observar que a quantidade de comentários neutros é maior que os outros dois, pode-se interpretar que essa métrica é ruim para os dados, e com isso podemos chegar em duas hipóteses: 1. 43% dos comentários não causam nenhum tipo de sentimento para as pessoas; ou 2. A classificação feita está equivocada, caso os posts causem algum tipo de sentimento. Além disso, a quantidade de comentários positivos é quase o dobro do negativo, o que se pode referir que os usuários estão se sentindo contentes com os serviços prestados. 

### 6.3.2 Pré - processamento
Abaixo serão descritos cada etapa do pré - processamento.

##### 6.3.2.1 Tratamento dos dados

&emsp;&emsp; Um dos primeiros tratamentos de dados que foi utilizado, foi o tratamento que retira as aspas duplas (“”) dos nomes das colunas da base de dados, já que anteriormente as colunas estavam da seguinte forma: “texto”, após esse tratamento, ficou apenas texto, como demonstra o código abaixo:
```
data = data.rename(columns={'"anomalia"' : 'anomalia', '"dataPublicada"' : 'dataPublicada', '"autor"' : 'autor', '"texto"' : 'texto', '"sentimento"' : 'sentimento', '"tipoInteracao"' : 'tipoInteracao', '"probabilidadeAnomalia"' : 'probabilidadeAnomalia', '"linkPost"' : 'linkPost', '"processado"' : 'processado',  '"contemHyperlink"' : 'contemHyperlink' })
```
&emsp;&emsp; Esse tratamento facilita o trabalho de chamar os textos das colunas de uma maneira mais simples, sem a necessidade de ter que colocar aspas, podendo chamar o texto diretamente.
<br>
&emsp;&emsp; O segundo tratamento realizado utilizou a função `data.describe()`, e com isso foi possível identificar que a coluna “processado” não agrega valor, uma vez que todos os seus valores são iguais a zero. Portanto, essa coluna foi removida da base de dados. Abaixo é possível ver o output desta função. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/data_describe.jpg"> <br>

&emsp;&emsp; Com isso, foi possível descartar essa coluna utilizando a função `data.drop()`. Vale ressaltar que as colunas “id” e “dataPublicada” também foram removidas da base de dados, uma vez que não possuem tanta relevância para uma análise de sentimento que tem como principal embasamento os textos, como mostra o código abaixo.
```
data_dropado = data.drop(['processado', 'id', 'dataPublicada'], axis=1)
data_dropado.head(3)
```
&emsp;&emsp; O terceiro tratamento realizado foi a remoção do autor @btgpactual, da coluna “autor”, foi possível removê-lo através de uma função que remove apenas o autor mencionado.
```
data_limpo = data_dropado.loc[data_dropado['autor'] != 'btgpactual']
data_limpo
```
&emsp;&emsp; Esse tratamento é necessário para o projeto, pois os comentários vindos desse autor não são tão relevantes para análise de sentimentos, uma vez que a maioria são respostas a comentários ou legendas dos posts.

##### 6.3.2.2 Tokenização

&emsp;&emsp;Para começar o pré - processamento pensando no modelo de análise de sentimento, é necessário separar as palavras dos textos em tokens, e o código abaixo define a função necessária para realizar esse processo.
```
def tokenizer(comment):
    if isinstance(comment, str):
        tokens = nltk.word_tokenize(comment)
        return tokens
    else:
        return []
```
&emsp;&emsp; A função acima realiza o processo descrito referenciando a biblioteca `nltk.word_tokenize`.

##### 6.3.2.3 Tratamento de abreviações

&emsp;&emsp;Para tornar mais fácil a análise de sentimento, foi feito um tratamento de abreviações, para que palavras como: “vcs” se torne “vocês”. O código abaixo define um dicionário de gírias e abreviações usado para normalização de texto.
```
# Dicionário de gírias e abreviações para normalização
dicionario_girias = {'vc': 'você', 'vcs':'você', 'Vc': 'você', 'pq': 'porque', 'Pq': 'porque', 'tbm': 'também', 'q': 'que', 'td': 'tudo', 'blz': 'beleza', 'flw': 'falou', 'kd': 'cadê', 'Gnt': 'gente', 'gnt': 'gente', 'to': 'estou', 'mt': 'muito', 'cmg': 'comigo', 'ctz': 'certeza', 'jah': 'já', 'naum': 'não', 'ta': 'está', 'eh': 'é', 'vdd': 'verdade', 'vlw': 'valeu', 'p': 'para', 'sdds': 'saudades', 'qnd': 'quando', 'msm': 'mesmo', 'fzr': 'fazer', 'ss': 'sim', 'Ss': 'sim', 'pdc': 'pode crer', 'nn': 'não', 'Nn': 'não', 'pls': 'please', 'obg': 'obrigado', 'agr': 'agora'}
```
&emsp;&emsp; O código abaixo apresenta um conjunto de palavras que são desconsideradas ou excluídas durante o processo. Essas palavras foram consideradas irrelevantes para a análise, já que contém informações específicas.

```
palavras_desconsideradas = {"warrenbrasil", "bicharaemotta", "sportainment", "sportainmet", "sportainmetâ", "sportainmentâ","roundpushpin",
"hubstage", "kaletsky", "scandiuzzi", "futofmoney", "ricktolledo", "thaotinhasbfc", "winthegame", "romulofialdini", "disclaimer", "astraoficialbr", "furnasenergia", "alelobrasil", "bancodaycoval", "grupohagana", "robertoljustus", "steinwaybrasil", "joseavillez", "dianaroth", "beachtennis", "alliancejjteam", "fabiogurgel", "blackrocks", "masterjacare", "gigipaivabjj", "clubefiinews", "mouratoglou", "octocapitalbr", "oficinadofraja" "blackintech"}
```

&emsp;&emsp; A função comentarios_normalizados abaixo tem como objetivo normalizar os comentários representados por tokens, levando em consideração o dicionário de gírias e abreviações e um conjunto de palavras desconsideradas, os dois citados acima. 

```
def comentarios_normalizados(tokens, dicionario_girias, palavras_desconsideradas):
  tokens_normalizados = []
  for sentence in tokens:
    treated = []
    for palavra in sentence:
        if palavra in palavras_desconsideradas:
            treated.append(palavra)
        else:
            if palavra in dicionario_girias:
                palavra_normalizada = dicionario_girias.get(palavra, palavra)
                treated.append(palavra_normalizada)
            else:
                treated.append(palavra)
    treated = [palavra.replace(' ', '') if '_' in palavra else palavra for palavra in treated]
    tokens_normalizados.append(treated)
  return tokens_normalizados
```

##### 6.3.2.4 Tratamento de emoji

&emsp;&emsp; A função demojize_tokens recebe uma lista de tokens e tem como objetivo realizar o processo de "demojize", ou seja, remover emojis e substituí-los por sua representação textual.
```
def demojize_tokens(tokens):
  demojized_tokens = []
  for termo in tokens:
    demojized = [emoji.demojize(palavra) if emoji.emoji_count(palavra) > 0 else palavra for palavra in termo]
    demojized = [palavra.replace(":", "").replace("_", "") if any(c in palavra for c in [":", "_"]) else palavra for palavra in demojized]
    demojized = [palavra.replace("-", "_") if "-" in palavra else palavra for palavra in demojized]
    demojized_tokens.append(demojized)
  return demojized_tokens
```

##### 6.3.2.5 Remoção de stopwords

&emsp;&emsp; Já que as palavras que são consideradas como _stopwords_ não tem uma importância para o sentido do texto e elas ocupam a maior parte dos tokens, essa etapa foi realizada por meio do código abaixo:
```
def remove_stopwords(tokens):
  stopwords = nltk.corpus.stopwords.words('portuguese')
  filtered_tokens = []
  for sentence in tokens:
      filtered = [palavra for palavra in sentence if palavra not in stopwords]
      filtered_tokens.append(filtered)
  return filtered_tokens
```
&emsp;&emsp; A função acima referencia a biblioteca para que as palavras classificadas sejam removidas do conjunto de _tokens_. 

##### 6.3.2.6 Remoção de alfanuméricos

&emsp;&emsp; Da mesma forma que algumas palavras não têm importância para a análise, a pontuação e caracteres especiais também não tem, por isso a função abaixo retira esses caracteres. 
```
def removendo_alfanumericos(tokens):
  output_tokens = []
  for sentence in tokens:
      output_list = []
      for palavra in sentence:
          if palavra.strip(): # Verifica se a palavra não é uma string vazia
              output_list.extend(re.findall(r'\w+', palavra)) # analisar se não é melhor usar o append em vez de extend
      output_tokens.append(output_list)
  return output_tokens
```

##### 6.3.2.7 Lematização

&emsp;&emsp; A função lematizacao() tem como objetivo realizar a lematização dos tokens, ou seja, converter as palavras para sua forma base ou lemma. O código utiliza o modelo pré-treinado do SpaCy para o idioma português (carregado anteriormente com o spacy.load("pt_core_news_sm")) para realizar a lematização.
```
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
```

##### 6.3.2.8 Pipeline

&emsp;&emsp; No pipeline foi dividido cada uma das funções em células separadas e depois é executado todas na ordem correta. Essa etapa permite que as funções sejam executadas na ordem correta, garantindo a consistência e a precisão dos resultados, e caso a ordem precise mudar, é mais simples fazer a alteração, essa organização torna o processo mais simples de entender e escalável.
<br>
&emsp;&emsp; Na parte de definição de funções, foi definida as funções que serão usadas no pipeline. As funções em questão são: tokenizer(); comentarios_normalizados(); demojize_tokens(); remove_stopwords(); removendo_alfanumericos(); lematizacao(), por fim, a função pipeline() executa cada uma das funções em ordem. Como as funções já foram apresentadas anteriormente, a seguir será mostrada a função:
```
def pipeline(comment):
	# Tokenização
      tokens = tokenizer(comment)
      # Normalização das abreviações
 	normalizado = comentarios_normalizados(tokens, dicionario_girias, palavras_desconsideradas)
      # Tratamento de Emojis
      demojized = demojize_tokens(normalizado)
      # Remoção das stopwords
      no_stopwords = remove_stopwords(tokens)
      # Remoção dos alfanuméricos
      no_alfanumericos = removendo_alfanumericos(no_stopwords)
      # lematização
      tratados = lematizacao(no_alfanumericos)
      return tratados

```
&emsp;&emsp; Por fim, foram realizados alguns testes de função para garantir que o fluxo do _pipeline_ estava operando adequadamente, para isso, foi criado um novo _dataframe_ com uma coluna chamada ‘pós_tratamento’, na qual está o resultado de todos os textos após passar pela função `pipeline()`. 
```
data_limpo[‘pós_tratamento’] = data_limpo['texto'].apply(pipeline)
```
&emsp;&emsp; A imagem abaixo exemplifica todos os processos descritos acima e conta com exemplos de inputs e outputs.

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/pipeline.jpg"> <br>

### 6.3.3 Modelo Bag of words

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/bow.jpg"> <br>

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
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/output.jpg"> <br>

&emsp;&emsp; Abaixo é demonstrado um exemplo resultante desta tabela, a qual possui um total de 12.193 linhas, que estão de acordo com cada comentário do csv disponibilizado pelo cliente, além de 24.331 colunas, que foram as palavras chaves selecionadas.
```
df['conf'].value_counts() 
0    11795
1      396
2        2
Name: conf, dtype: int64
```
&emsp;&emsp; Neste exemplo, é possível perceber que o termo `‘conf’`  se repete uma vez, em 396 comentários diferentes, e se repete duas vezes em 2 comentários diferentes. Dessa forma, percebe-se como a função consegue selecionar palavras chaves que estão contidas nas diversas frases do dataframe.

## 6.4 Conclusão

### 6.4.1 Análise descritiva
&emsp;&emsp; Esta análise descritiva dos gráficos proporciona uma compreensão mais profunda dos dados, permitindo identificar _insights_ e tomar decisões. É importante ressaltar que as conclusões obtidas são interpretadas considerando o contexto específico dos dados e as questões de pesquisa em análise.
### 6.4.2 Pré - processamento
&emsp;&emsp; O pré-processamento dos dados é fundamental para garantir a qualidade e a confiabilidade das análises posteriores, contribuindo para um melhor entendimento dos dados e para a obtenção de resultados mais precisos e significativos.
### 6.4.3 Modelo Bag of words
&emsp;&emsp; Com a aplicação do Modelo _Bag of Words (BoW)_ é possível perceber a capacidade de seleção de palavras para a futura implementação na _Machine Learning_ desenvolvida. O objetivo do projeto é demonstrado a partir da imagem abaixo:

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/modelo.jpg"> <br>

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

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/nuvem_palavras.png"> <br>

&emsp;&emsp; Assim, o próximo passo é um retratamento dos textos para ter melhor desenvolvimento e aplicação no momento de construção da Inteligência Artificial.
