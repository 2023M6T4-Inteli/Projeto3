# **10. Comparação entre modelos**

## 10.1 Introdução

&emsp;&emsp; Nesta análise, foram utilizados dois modelos de processamento de linguagem natural (PLN) utilizados para classificação de texto. O primeiro modelo é uma combinação do algoritmo Naive Bayes com o método Word2Vec usando a arquitetura Continuous Bag-of-Words (CBOW), que obteve uma precisão de 43,55%. O segundo modelo consiste apenas no método Word2Vec aplicado diretamente ao corpus, alcançando uma precisão de 42,77%. A seguir será apresentado todos os métodos utilizados, os resultados obtidos e ao final uma conclusão sobre a comparação entre esses dois modelos.

## 10.2 Método
&emsp;&emsp; O primeiro modelo empregou uma abordagem híbrida, combinando o algoritmo Naive Bayes com o método Word2Vec usando a arquitetura CBOW. O Naive Bayes é um algoritmo de classificação probabilístico amplamente utilizado, enquanto o Word2Vec é uma técnica de aprendizado de representação de palavras baseada em redes neurais. <br>
&emsp;&emsp; No caso deste modelo, o CBOW foi utilizado para gerar representações vetoriais de palavras, que foram então alimentadas ao classificador Naive Bayes para realizar a classificação. <br>
&emsp;&emsp; Já o segundo modelo adotou apenas o método Word2Vec, sem a inclusão do Naive Bayes. Nessa abordagem, o Word2Vec foi aplicado diretamente ao corpus de texto para aprender representações vetoriais de palavras.<br>

## 10.3 Resultados

&emsp;&emsp; O primeiro modelo, que combina Naive Bayes e Word2Vec com CBOW, obteve uma precisão de 43,55%. Já o segundo modelo, que utiliza apenas o Word2Vec no corpus, alcançou uma precisão de 42,77%.


### 10.4 Conclusão
&emsp;&emsp; Ao comparar os dois modelos, foi possível observar que o primeiro modelo, que combina Naive Bayes e Word2Vec com CBOW, apresentou uma precisão com 0,78% a mais de diferença, em relação ao segundo modelo que utiliza apenas o Word2Vec no corpus. No entanto, a diferença entre as precisões é relativamente pequena, o que sugere que ambos os modelos possuem desempenho semelhante na tarefa de classificação de texto. <br>
&emsp;&emsp; Vale ressaltar que é importante considerar outros aspectos além da precisão ao avaliar a adequação de um modelo para uma tarefa específica. A simplicidade e a eficiência computacional são fatores relevantes a serem considerados. O primeiro modelo, que combina o Naive Bayes com Word2Vec, exigiu um processamento mais complexo e potencialmente maior tempo de treinamento e inferência em comparação ao segundo modelo que utilizava apenas o Word2Vec. Portanto, é necessário levar em consideração os métodos de cada modelo, para que assim possa-se escolher o modelo mais adequado para a aplicação do projeto, que tem como objetivo uma análise de sentimento e identificação de palavras-chave utilizando PLN. <br>
&emsp;&emsp; Sendo assim, embora o modelo Naive Bayes + Word2Vec com CBOW tenha apresentado uma vantagem, mesmo que mínima, em termos de precisão em relação ao modelo Word2Vec no corpus, a diferença não é significativa. A escolha entre esses modelos dependerá de outros fatores, como eficiência computacional e requisitos específicos do projeto em si.  <br>







