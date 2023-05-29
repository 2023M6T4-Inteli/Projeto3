# **10. Criação do Modelo - Word2Vec**
## 2. Entendimento e Tratamento dos Dados
Foi criado um novo Data Frame contendo informações valiosas sobre os comentários de usuários nas redes sociais, permitindo identificar áreas 
que precisam de melhorias e atender às necessidades dos usuários. A análise de sentimento pode ser usada para identificar tendências e
padrões na percepção dos usuários ao longo do tempo. Para isso, separamos o Data Frame em três colunas; A coluna "autor" indica quem fez 
cada comentário, permitindo que os proprietários da plataforma identifiquem os usuários mais engajados e atendam às suas necessidades. Já a 
coluna de "sentimento" varia entre 0, 1 e 2 , no qual 0 é para negativo, 1 para neutro e 2 para positivo, essa coluna serve para entender a 
percepção dos usuários, com base no sentimento expresso pelo usuário; A coluna "Texto_tratado" é particularmente útil, pois ela contém os
comentários após o pré-processamento, facilitando a análise de sentimento. Em resumo, o DataFrame é uma ferramenta valiosa para a análise 
de sentimento e mineração de opiniões na plataforma online.

## 3. Bag of Words (BoW)
O modelo Bag of Words (BoW) é uma técnica utilizada em processamento de linguagem natural para representar um texto como um conjunto de
palavras desordenadas, ignorando a ordem e a estrutura gramatical das frases. Nesse modelo, cada palavra única do texto é transformada em
uma "feature" (característica), e a frequência de cada palavra no texto é usada como um valor numérico para a feature correspondente. Por
exemplo, a frase "O gato preto pulou o muro" seria representada como um conjunto de palavras desordenadas: 'o', 'gato', 'preto', 'pulou', 
'o', 'muro'. A frequência de cada palavra seria contada, e o resultado seria um vetor numérico que representa a frequência de cada palavra
na frase.

## 10. Word2Vec com CBOW
O modelo Word2Vec com CBOW é uma arquitetura do modelo Word2Vec que tenta prever a palavra central a partir do contexto de palavras ao seu
redor. Durante o treinamento, a rede neural aprende a associar as palavras com vetores de números reais que capturam as relações semânticas
e sintáticas entre elas.

## 11. Naive Bayes + Word2Vec com CBOW
O modelo Naive Bayes é um algoritmo de classificação probabilística que utiliza a representação vetorial de cada palavra no documento para
estimar a probabilidade de pertencer a uma classe. No modelo Naive Bayes com Word2Vec ou CBOW, a representação vetorial de cada palavra é 
obtida a partir de modelos treinados previamente. O modelo Naive Bayes é treinado com esses vetores de documentos para estimar a 
probabilidade de pertencer a cada classe.

## 12. Word2Vec com embedding layer
O modelo Word2Vec com embedding layer é uma rede neural que utiliza uma camada de embedding para aprender a representação vetorial de cada
palavra em um corpus de texto. Essa representação é usada para prever a palavra seguinte em uma sequência de palavras, e o modelo é treinado
para minimizar o erro entre a palavra prevista e a palavra real. O modelo resultante pode ser usado para encontrar palavras semanticamente
similares.

## 13. Naive Bayes + Word2Vec com embedding layer
O modelo Naive Bayes com Word2Vec ou embedding layer é um algoritmo de classificação que utiliza a representação vetorial de cada palavra
em um documento. Essa representação é obtida a partir de modelos treinados previamente e usada como características para estimar a 
probabilidade de pertencer a uma classe. O modelo é treinado com esses vetores de documentos para fazer inferências sobre a probabilidade
de um documento pertencer a uma classe.

## 14. Resultados
Abaixo é possível ver os resultados finais do modelo. Temos quatro métricas para medir o sucesso do modelo: precisão, revocação, f1-score
e suporte. No final atingimos uma acurácia de incríveis 43%.
```
                    precision    recall  f1-score   support

                  0       0.30      0.77      0.43       417
                  1       0.77      0.45      0.57       855
                  2       0.34      0.17      0.22       624

       Accuracy                                0.43      1896
    macro avg       0.47      0.46      0.41      1896
weighted avg       0.52      0.43      0.42      1896
```
## 15. Comparação BoW e Word2Vec
  Bag-of-Words (BoW) e Word2Vec são técnicas de PLN para representar o texto em formato numérico. BoW considera a frequência de cada
  palavra no texto, enquanto Word2Vec aprende relações semânticas entre palavras. Word2Vec é mais eficaz em lidar com sinônimos, antônimos
  e relações semânticas, e gera vetores de dimensão mais baixa do que BoW. BoW é melhor para tarefas de classificação simples, enquanto 
  Word2Vec é mais adequado para tarefas em que o significado e o contexto das palavras são importantes, como análise de sentimento, 
  tradução automática e sistemas de recomendação de conteúdo.

