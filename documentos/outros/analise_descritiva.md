# **6. Análise Descritiva**

## 6.1 Introdução
&emsp;&emsp; A análise descritiva é uma técnica estatística que pode ser aplicada em diferentes áreas, incluindo a análise de sentimentos. No contexto do projeto proposto pelo BTG de análise de sentimentos realizado a partir de comentários de usuários em publicações no _Instagram_ do banco, a análise descritiva é utilizada para descrever e resumir as principais características dos dados coletados. <br>
&emsp;&emsp; Por meio desta, é possível obter informações sobre: 
- O número total de comentários coletados; </br>
- A distribuição de sentimentos positivos, negativos e neutros expressos pelos usuários; </br>
- As palavras mais frequentes nos comentários; </br>
- Os usuários que mais realizaram comentários; </br>

&emsp;&emsp; Essas informações são cruciais para compreender melhor a percepção dos usuários em relação ao banco e para orientar futuras estratégias de comunicação e relacionamento com o público, garantindo uma maior assertividade em futuras publicações do banco BTG.

## 6.2 Método
&emsp;&emsp; Os comentários realizados pelos usuários nas publicações do banco BTG são uma fonte valiosa de informações para entender como os clientes se sentem em relação aos serviços oferecidos pela instituição financeira. Para realizar a análise desses dados, foram utilizados diversos métodos de tratamento de dados, que serão descritos no tópico 6.2.2 Pré - processamento. </br>
&emsp;&emsp; Para visualizar as informações de maneira clara e acessível, foram utilizados gráficos de barra e pizza. Os gráficos de barra foram utilizados para a visualização das palavras mais frequentes encontradas nos comentários; identificação dos autores mais ativos e o tipo de sentimento causado (positivo, neutro ou negativo). Já nos gráficos de pizza destacam os tipos de interação mais utilizados, alternando entre comentários, menções e respostas e também tipos de sentimentos expressos pelos usuários. </br>
&emsp;&emsp; Para realizar a análise e a visualização desses dados, foram utilizadas bibliotecas como: _Matplotlib_, que é uma biblioteca de visualização de dados em _Python_, além de bibliotecas notáveis como é o caso do _pandas_, _numpy_ e a _nltk_. Com essas ferramentas, foi possível obter _insights_ valiosos sobre a percepção dos clientes em relação ao Banco BTG e identificar áreas que precisam de melhorias.

## 6.3 Resultados
&emsp;&emsp; Na análise descritiva dos dados, foram explorados 2 tipos de gráficos: gráficos de pizza e barras. Utilizando técnicas de visualização, foi possível apresentar informações relevantes e obter _insights_ sobre os dados em questão. Abaixo serão descritos os gráficos e os resultados obtidos. 

##### 6.3.1 Autores
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

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/autores.png"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 20: Gráfico “Autores que mais comentam”
<br>

&emsp;&emsp; Com esse gráfico foi possível observar que, na maioria das vezes, tem um padrão muito claro de frequência de comentários, o que significa que a empresa mantém um público específico que também é muito engajado. Apesar disso, foi criada a hipótese de que, pelo fato do primeiro usuário (@amgcapitalinvest) ser uma empresa credenciada pelo BTG, ela marca o banco nos seus posts, referenciando os créditos, é possível interpretar que talvez não sejam somente comentários. 
<br>

##### 6.3.2 Palavras mais frequentes
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

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/palavras.png"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 21: Gráfico “Top 20 palavras mais frequentes”
<br>

&emsp;&emsp; Com esse gráfico é possível observar que entre as 20 palavras, 7 delas estão diretamente relacionadas ao banco: “btgpactual”, “invest”, “btg”, “banc”, “merc”, “financeir”, “pactual”. Isso pode significar que geralmente as pessoas estão respondendo o post com os assuntos neles descritos, que, na maioria das vezes, tem como tema o mercado financeiro. Além disso, foi criada uma hipótese que a palavra “btgpactual” se diz respeito à marcação da conta do banco e não necessariamente falando sobre ele, já que as palavras: “btg” e “pactual” estão entre as 20 palavras mais frequentes. 
<br>

##### 6.3.3 Tipos de interação
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

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/interacao.png">
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 22: Gráfico “Tipos de interação”
<br>

&emsp;&emsp; O gráfico acima demonstra que, caso a hipótese de ter repost dos posts do BTG esteja certa, o dataframe está, em sua maioria com esses casos, o que torna preocupante, já que a ideia é que o projeto analise comentários dos posts. Além disso, pode-se observar uma diferença significativa entre “comentários” e “resposta”. 

##### 6.3.4 Classificação de sentimento
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
<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_pizza.png"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 23: Gráfico “Tipos de sentimento” - Pizza
<br>
**O segundo gráfico gerado é o de barras**:
<br>
&emsp;&emsp; A primeira linha do gráfico de barra define qual será o tamanho da figura que será gerada no final do código. A seguir, a variável `count_sentimentos` é plotada utilizando o método `plot` com o parâmetro `kind='bar'`, indicando o tipo de gráfico, essa linha que diferencia os tipos de gráficos. As próximas 3 linhas são usadas para definir os rótulos dos eixo x e y e o título do gráfico. A última linha exibe o gráfico a seguir na saída. 

<img src="https://github.com/2023M6T4-Inteli/Projeto3/blob/main/assets/imagens/sentimento_coluna.png"> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Figura 24: Gráfico “Tipos de sentimento” - Barras
<br>
&emsp;&emsp; Analisando os gráficos é possível observar que a quantidade de comentários neutros é maior que os outros dois, pode-se interpretar que essa métrica é ruim para os dados, e com isso podemos chegar em duas hipóteses: 1. 43% dos comentários não causam nenhum tipo de sentimento para as pessoas; ou 2. A classificação feita está equivocada, caso os posts causem algum tipo de sentimento. Além disso, a quantidade de comentários positivos é quase o dobro do negativo, o que se pode referir que os usuários estão se sentindo contentes com os serviços prestados. 

## 6.4 Conclusão
&emsp;&emsp; Esta análise descritiva dos gráficos proporciona uma compreensão mais profunda dos dados, permitindo identificar _insights_ e tomar decisões. É importante ressaltar que as conclusões obtidas são interpretadas considerando o contexto específico dos dados e as questões de pesquisa em análise.
