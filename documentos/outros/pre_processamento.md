# **7. Pré - processamento**

## 7.1 Introdução
&emsp;&emsp; O pré-processamento de dados no contexto do PLN refere-se a uma série de etapas de preparação que os dados textuais devem passar antes de serem usados em um modelo de aprendizado de máquina. Essas etapas visam limpar, organizar e estruturar os dados textuais para que sejam mais facilmente compreendidos pelo modelo. Algumas etapas importantes do pré- processamento são: 
- Tokenização; </br> 
- Tratamento de abreviações; </br> 
- Tratamento de emoji; </br> 
- Remoção de stopwords; </br>
- Remoção de alfanuméricos; </br>
- Lematização; </br> 

&emsp;&emsp; Além disso, foi realizado um tratamento dos dados e a definição de uma função _pipeline_.

## 7.2 Método
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

## 7.3 Resultados
Abaixo serão descritos cada etapa do pré - processamento.

##### 7.3.1 Tratamento dos dados

&emsp;&emsp; Um dos primeiros tratamentos de dados que foi utilizado, foi o tratamento que retira as aspas duplas (“”) dos nomes das colunas da base de dados, já que anteriormente as colunas estavam da seguinte forma: “texto”, após esse tratamento, ficou apenas texto, como demonstra o código abaixo:
```
data = data.rename(columns={'"anomalia"' : 'anomalia', '"dataPublicada"' : 'dataPublicada', '"autor"' : 'autor', '"texto"' : 'texto', '"sentimento"' : 'sentimento', '"tipoInteracao"' : 'tipoInteracao', '"probabilidadeAnomalia"' : 'probabilidadeAnomalia', '"linkPost"' : 'linkPost', '"processado"' : 'processado',  '"contemHyperlink"' : 'contemHyperlink' })
```
&emsp;&emsp; Esse tratamento facilita o trabalho de chamar os textos das colunas de uma maneira mais simples, sem a necessidade de ter que colocar aspas, podendo chamar o texto diretamente.
<br>
&emsp;&emsp; O segundo tratamento realizado utilizou a função `data.describe()`, e com isso foi possível identificar que a coluna “processado” não agrega valor, uma vez que todos os seus valores são iguais a zero. Portanto, essa coluna foi removida da base de dados. Abaixo é possível ver o output desta função. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/data_describe.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 25: Output da função data.describe()
<br>

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

##### 7.3.2 Tokenização

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

##### 7.3.3 Tratamento de abreviações

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

##### 7.3.4 Tratamento de emoji

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

##### 7.3.5 Remoção de stopwords

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

##### 7.3.6 Remoção de alfanuméricos

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

##### 7.3.7 Lematização

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

##### 7.3.8 Pipeline

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

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/pipeline.jpg">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 26: Demonstração do pipeline
<br>

## 7.4 Conclusão
&emsp;&emsp; O pré-processamento dos dados é fundamental para garantir a qualidade e a confiabilidade das análises posteriores, contribuindo para um melhor entendimento dos dados e para a obtenção de resultados mais precisos e significativos.
