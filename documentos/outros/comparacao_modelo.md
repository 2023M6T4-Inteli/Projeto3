# **10. Comparação entre modelos**

## 10.1 Introdução

&emsp;&emsp; Essa comparação é feita com os modelos desenvolvidos durante a Sprint 4, que foram realizados com base com pré-processamento de duas formas diferentes: sprint 3 e sprint 4. Por isso, os tópicos abaixo serão divididos com os nomes dos modelos e será comparado o resultado de recall ou acurácia, a depender do modelo. 

## 10.2 Método
&emsp;&emsp; 	O primeiro modelo comparado é o Naive Bayes com BoW, usando os métodos de Grid Search e Cross Validation. Com isso, o modelo e seus métodos foram aplicados à base da Sprint 3 e da Sprint 4. <br>
&emsp;&emsp; 	O segundo modelo comparado é o Random Forest com BoW, usando os métodos de Grid Search e Cross Validation. Com isso, o modelo e seus métodos foram aplicados à base da Sprint 3 e da Sprint 4.  <br>
&emsp;&emsp; 	O terceiro modelo comparado é o Rede Neural - Sequência de palavras. Com isso, o modelo foi aplicado à base da Sprint 3 e da Sprint 4, e com o respectivos modelos de vetorização: Word2Vec + CBoW e Word2Vec + Embedding Layer.<br>
&emsp;&emsp; 	O quarto modelo comparado é o Rede Neural com Word2Vec. Com isso, o modelo foi aplicado à base da Sprint 3 e da Sprint 4.  <br>
&emsp;&emsp; 	O quinto modelo comparado é o Random Forest com Word2Vec. Com isso, o modelo foi aplicado à base da Sprint 3 e da Sprint 4. <br>

## 10.3 Resultados

&emsp;&emsp; Serão apresentados os resultados de todos os modelos apresentados acima. Os tópicos serão divididos com o nome dos modelos.

### 10.3.1 Naive Bayes com BoW 
&emsp;&emsp; O modelo Naive Bayes sem nenhum dos métodos, na Sprint 3, obteve os seguintes resultados:

```
             precision    recall  f1-score   support
           0       0.00      0.00      0.00         0
           1       0.88      0.46      0.61      1230
           2       0.63      0.76      0.69       612

    accuracy                           0.56      1842
   macro avg       0.50      0.41      0.43      1842
weighted avg       0.79      0.56      0.63      1842
```

&emsp;&emsp; Já o modelo Naive Bayes aplicado na base da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       0.85      0.30      0.45       957
           2       0.66      0.79      0.72       651

    accuracy                           0.50      1608
   macro avg       0.50      0.36      0.39      1608
weighted avg       0.77      0.50      0.56      1608
```
&emsp;&emsp; Analisando os resultados, é possível observar que, na Sprint 4, o modelo Naive Bayes apresenta um desempenho ruim em termos de recall para a classe 1 (neutro), com um valor de 30%. Para a classe 2 (positivo) o recall é mais alto na Sprint 4, com um valor de 79%. No entanto, é importante notar que o modelo não possui suporte para a classe 0 em ambas as sprints, o que limita a sua capacidade de generalização. <br>
&emsp;&emsp; Considerando a métrica de recall como a mais importante, os resultados indicam que o modelo Naive Bayes aplicado na base da Sprint 4 apresenta um melhor desempenho em relação ao recall, mas ainda sim é um modelo que tem limitações por conta do valor 0 na classe 0 (negativo).

### 10.3.2 Naive Bayes - Cross Validation
&emsp;&emsp; O modelo Naive Bayes com o método de Cross Validation, na Sprint 3, obteve os seguintes resultados:

```
             precision    recall  f1-score   support

           0       0.63      0.73      0.68      1974
           1       0.78      0.58      0.67      4012
           2       0.62      0.75      0.68      3221

    accuracy                           0.67      9207
   macro avg       0.67      0.69      0.67      9207
weighted avg       0.69      0.67      0.67      9207
```

&emsp;&emsp; O modelo Naive Bayes com o método de Cross Validation, na Sprint 4, obteve os seguintes resultados:

```
             precision    recall  f1-score   support

           0       0.58      0.75      0.65      1970
           1       0.73      0.36      0.48      2918
           2       0.62      0.79      0.70      3152

    accuracy                           0.63      8040
   macro avg       0.64      0.64      0.61      8040
weighted avg       0.65      0.63      0.61      8040
```

&emsp;&emsp; Comparando esses resultados, pode-se observar que, em ambas as sprints, o modelo Naive Bayes apresenta um recall mais elevado para a classe 2 (positivo) em comparação com as outras classes. Isso indica que o modelo tem um desempenho melhor em identificar corretamente os exemplos da classe 2 em comparação com as outras duas classes. No entanto, em termos gerais, o modelo Naive Bayes apresenta resultados mais consistentes na Sprint 3 em comparação com a Sprint 4, principalmente em relação ao recall para a classe 1 (neutro), onde o recall para a classe 1 foi de 58% e 36%, respectivamente. No geral, o modelo aplicado com a base da Sprint 3 apresentou um desempenho melhor, levando em consideração que o recall é a métrica mais importante.

### 10.3.3 Naive Bayes - Grid Search
&emsp;&emsp; Abaixo serão analisados os resultados em termos de melhores hiperparâmetros encontrados, melhor acurácia no conjunto de teste e melhor revocação. Esses resultados levam em consideração a base desenvolvida durante a Sprint 3 e Sprint 4.

```
Sprint 3:
Melhores hiperparâmetros: {'alpha': 0.5, 'fit_prior': True}
Melhor acurácia no conjunto de teste: 0.5537459283387622
Melhor revocação: 0.7103233123310607

	Sprint 4:
Melhores hiperparâmetros: {'alpha': 1.0, 'fit_prior': True}
Melhor acurácia no conjunto de teste: 0.4993781094527363
Melhor revocação: 0.6609166980005126

```

&emsp;&emsp; Em relação aos melhores hiperparâmetros, a Sprint 3 encontrou um valor de alpha igual a 0.5, enquanto a Sprint 4 encontrou um valor de alpha igual a 1.0. Ambas as sprints encontraram o mesmo valor para o hiperparâmetro fit_prior, que é True. No que diz respeito à melhor acurácia no conjunto de teste, a Sprint 3 obteve uma acurácia de aproximadamente 0.554 (55,4%), enquanto a Sprint 4 obteve uma acurácia de aproximadamente 0.499 (49,9%). Já a revocação, a Sprint 3 obteve um valor de cerca de 0.710 (71%), enquanto a Sprint 4 obteve um valor de cerca de 0.661 (66%). <br>
&emsp;&emsp; A Sprint 3 teve um desempenho um pouco superior em relação à acurácia e revocação quando comparada com a Sprint 4. Por isso, pode-se concluir que o modelo com a base da Sprint 3, teve um melhor desempenho em um geral.

### 10.3.4 Random Forest + BoW
&emsp;&emsp; O modelo Random Forest sem nenhum dos métodos, na Sprint 3, obteve os seguintes resultados:

```
            precision    recall  f1-score   support

           0       0.69      0.50      0.58       386
           1       0.76      0.74      0.75       844
           2       0.64      0.77      0.70       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```

&emsp;&emsp; Já o modelo Random Forest aplicado na base da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.72      0.50      0.59       360
           1       0.65      0.68      0.66       597
           2       0.68      0.76      0.72       651

    accuracy                           0.67      1608
   macro avg       0.68      0.65      0.66      1608
weighted avg       0.68      0.67      0.67      1608
```

&emsp;&emsp; Comparando os resultados dos modelos Random Forest que tem aplicações com a base da Sprint 3 e da Sprint 4, pode-se observar que o desempenho é bastante similar nas duas sprints. Para a classe 0 (negativo), o recall é o mesmo nas duas sprints (50%). Para a classe 1 (neutro), houve uma ligeira queda no recall na Sprint 4 (68%) em comparação com a Sprint 3 (74%). Para a classe 2 (positivo), houve uma ligeira queda no recall na Sprint 4 (76%) em relação à Sprint 3 (77%). Mas com base nessa análise, não é possível determinar de forma definitiva qual modelo é o melhor em termos de recall, pois ambos apresentam resultados bastante similares nas duas sprints, já que o desempenho do modelo é relativamente consistente. Essa conclusão é feita considerando que o recall foi a métrica escolhida pelo grupo. 

### 10.3.5 Random Forest - Cross Validation
&emsp;&emsp; O modelo Random Forest com o método de Cross Validation, na Sprint 3, obteve os seguintes resultados:

```
             precision    recall  f1-score   support

           0       0.71      0.50      0.58       386
           1       0.77      0.75      0.76       844
           2       0.63      0.77      0.69       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```

&emsp;&emsp; Já o modelo Random Forest com o método de Cross Validation aplicado na base da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.74      0.51      0.60       360
           1       0.65      0.70      0.68       597
           2       0.70      0.77      0.73       651

    accuracy                           0.69      1608
   macro avg       0.70      0.66      0.67      1608
weighted avg       0.69      0.69      0.68      1608
```

&emsp;&emsp; Ao analisar os resultados em texto corrido, é possível observar que o desempenho dos modelos é relativamente semelhante nas duas sprints. Ambos apresentam uma acurácia em torno de 0,70, indicando que os modelos têm uma taxa de acerto razoável. Comparando os resultados, podemos observar que, para a classe 0 (negativo), o modelo da Sprint 4 apresentou um ligeiro aumento no recall em comparação com o modelo da Sprint 3 (51% e 50%). Para a classe 1 (neutro), o modelo da Sprint 3 teve um recall ligeiramente maior (75% e 70%), enquanto que para a classe 2, ambos os modelos apresentaram o mesmo recall (77%). Porém as variações são muito sutis, então não é possível determinar a melhor aplicação do modelo, já que o grupo optou por levar em consideração a métrica de recall.

### 10.3.6 Random Forest - Grid Search
&emsp;&emsp; O modelo Random Forest com o método de Cross Validation aplicado na base da Sprint 3, obteve esses resultados: 

```
             precision    recall  f1-score   support

           0       0.69      0.50      0.58       386
           1       0.76      0.74      0.75       844
           2       0.64      0.77      0.70       612

    accuracy                           0.70      1842
   macro avg       0.70      0.67      0.68      1842
weighted avg       0.71      0.70      0.70      1842
```

&emsp;&emsp; O modelo Random Forest com o método de Cross Validation aplicado na base da Sprint 4, obteve esses resultados: 

```
             precision    recall  f1-score   support

           0       0.74      0.51      0.60       360
           1       0.65      0.70      0.68       597
           2       0.70      0.77      0.73       651

    accuracy                           0.69      1608
   macro avg       0.70      0.66      0.67      1608
weighted avg       0.69      0.69      0.68      1608
```

&emsp;&emsp; Comparando os dois relatórios de classificação, é possível observar que o desempenho é bastante semelhante, com algumas diferenças sutis. O modelo da Sprint 4 apresentou uma precisão um pouco mais alta para a classe 0 (negativo), enquanto o modelo da Sprint 3 teve uma precisão ligeiramente maior para as classes 1 (neutro) e 2 (positivo). Em relação ao recall e F1-score, o modelo da Sprint 4 teve resultados um pouco melhores para todas as classes. Como o grupo definiu a métrica recall como a mais importante, o modelo que foi desenvolvido com o pré processamento da Sprint 4 é o melhor, já que obteve resultados superiores.

### 10.3.7 Rede Neural (Sequência de palavras) - base tratada
&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base tratada da Sprint 3, obteve esses resultados:

```
              precision    recall  f1-score   support

           0       0.70      0.68      0.69       662
           1       0.77      0.76      0.77      1358
           2       0.71      0.73      0.72      1019

    accuracy                           0.73      3039
   macro avg       0.72      0.72      0.72      3039
weighted avg       0.73      0.73      0.73      3039
```

&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base tratada da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.67      0.66      0.67       624
           1       0.72      0.64      0.68       990
           2       0.69      0.78      0.73      1040

    accuracy                           0.70      2654
   macro avg       0.70      0.69      0.69      2654
weighted avg       0.70      0.70      0.70      2654
```

&emsp;&emsp; Analisando esses resultados, pode-se observar que na Sprint 3 a rede neural apresentou um desempenho um pouco melhor em termos de recall para as classes 0 (negativo) e 2 (positivo) em comparação com a Sprint 4. Para a classe 0, o recall diminuiu de 68% para 66%, e para a classe 2, o recall diminuiu de 73% para 78%. No entanto, para a classe 1 (neutro), houve uma ligeira melhoria no recall na Sprint 4, passando de 76% para 64%. Considerando a métrica de recall como a mais importante, podemos concluir que a rede neural teve um desempenho um pouco melhor na Sprint 3, especialmente para as classes 0 e 2.

### 10.3.8 Rede Neural (Sequência de palavras) - Word2Vec + CBoW
&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base com o Word2Vec e CBoW da Sprint 3, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.70      0.70      0.70       633
           1       0.79      0.74      0.77      1308
           2       0.71      0.77      0.74      1098

    accuracy                           0.74      3039
   macro avg       0.73      0.74      0.73      3039
weighted avg       0.74      0.74      0.74      3039
```

&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base com o Word2Vec e CBoW da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.66      0.63      0.65       655
           1       0.71      0.64      0.68       969
           2       0.69      0.77      0.73      1030

    accuracy                           0.69      2654
   macro avg       0.69      0.68      0.68      2654
weighted avg       0.69      0.69      0.69      2654
```

&emsp;&emsp; Com esses resultados, pode-se observar que na Sprint 3 a rede neural apresentou um desempenho ligeiramente melhor em termos de recall para as classes 0 (negativo) e 1 (neutro) em comparação com a Sprint 4. Para a classe 0, o recall diminuiu de 70% para 63%, e para a classe 1, o recall diminuiu de 74% para 64%. No entanto, para a classe 2 (positivo), o recall se manteve estável em ambos os modelos, com um valor de 0,77. Já que o grupo definiu que a métrica de recall é a mais relevante, é possível concluir que a rede neural com o método de sequência de palavras aplicado na base com Word2Vec e CBoW teve um desempenho um pouco melhor na Sprint 3, especialmente para as classes 0 e 1. No entanto, o recall para a classe 2 se manteve consistente em ambos os modelos.

### 10.3.9 Rede Neural (Sequência de palavras) - Word2Vec + Embedding Layer
&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base com o Word2Vec e Embedding Layer da Sprint 3, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.67      0.69      0.68       632
           1       0.76      0.75      0.76      1321
           2       0.72      0.72      0.72      1086

    accuracy                           0.73      3039
   macro avg       0.72      0.72      0.72      3039
weighted avg       0.73      0.73      0.73      3039
```

&emsp;&emsp; O modelo Rede Neural com o método de sequência de palavra aplicado na base com o Word2Vec e Embedding Layer da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.68      0.64      0.66       676
           1       0.68      0.69      0.69       980
           2       0.70      0.72      0.71       998

    accuracy                           0.69      2654
   macro avg       0.69      0.68      0.69      2654
weighted avg       0.69      0.69      0.69      2654
```

&emsp;&emsp; Observando esses resultados, pode-se notar que na Sprint 3 a rede neural apresentou um desempenho ligeiramente melhor em termos de recall para as classes 0 (negativo) e 2 (positivo) em comparação com a Sprint 4. Para a classe 0, o recall diminuiu de 69% para 64%, enquanto para a classe 2 o recall se manteve em 72%. No entanto, para a classe 1 (neutro), a rede neural da Sprint 4 obteve um recall um pouco maior, diminuindo de 75% para 69% em comparação com a Sprint 3. <br>
&emsp;&emsp; Considerando a métrica de recall como a mais importante, podemos concluir que, embora a rede neural da Sprint 3 tenha apresentado um melhor desempenho para as classes 0 e 2, a rede neural da Sprint 4 teve um recall maior para a classe 1. Portanto, a escolha do melhor modelo dependerá da importância relativa das diferentes classes.

### 10.3.10 Rede Neural sem embedding - Word2Vec
&emsp;&emsp; O modelo Rede Neural com o Word2Vec da Sprint 3, obteve esses resultados:

```
                precision    recall  f1-score   support

           0       0.53      0.54      0.53       386
           1       0.69      0.65      0.67       844
           2       0.56      0.59      0.57       612

    accuracy                           0.61      1842
   macro avg       0.59      0.59      0.59      1842
weighted avg       0.61      0.61      0.61      1842
```

&emsp;&emsp; O modelo Rede Neural com o Word2Vec da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.40      0.62      0.49       360
           1       0.62      0.47      0.54       597
           2       0.58      0.54      0.56       651

    accuracy                           0.53      1608
   macro avg       0.53      0.54      0.53      1608
weighted avg       0.56      0.53      0.53      1608
```

&emsp;&emsp; Observando esses resultados, é possível ver que, em ambas as sprints, o recall para todas as classes não é muito alto. No entanto, na sprint 4, o recall para a classe 0 (negativo) melhorou significativamente em relação à sprint 3 (62% e 54%). O recall para a classe 1 (neutro) diminuiu (47% e 65%) e o recall para a classe 2 (positivo) manteve praticamente o mesmo (54% vs 59%). Considerando que o recall é a métrica mais importante, definida pelo grupo, é possível concluir que o modelo da sprint 4 tem um desempenho melhor em relação à classe 0, mas pior em relação à classe 1. Da mesma forma que o modelo passado, a escolha do melhor modelo dependerá da importância relativa das diferentes classes.

### 10.3.11 Random Forest - Word2Vec
&emsp;&emsp; O modelo Random Forest com o Word2Vec da Sprint 3, obteve esses resultados:

```
	         precision    recall  f1-score   support

           0       0.56      0.54      0.55       386
           1       0.75      0.70      0.73       844
           2       0.60      0.67      0.63       612

    accuracy                           0.66      1842
   macro avg       0.64      0.64      0.64      1842
weighted avg       0.66      0.66      0.66      1842
```

&emsp;&emsp; O modelo Random Forest com o Word2Vec da Sprint 4, obteve esses resultados:

```
             precision    recall  f1-score   support

           0       0.55      0.54      0.55       360
           1       0.66      0.57      0.61       597
           2       0.62      0.70      0.65       651

    accuracy                           0.61      1608
   macro avg       0.61      0.60      0.60      1608
weighted avg       0.62      0.61      0.61      1608
```

&emsp;&emsp; Com esses resultados apresentados acima, é possível ver que o recall para a classe 0 (negativo) é semelhante em ambos os modelos (54%). O recall para a classe 1 (neutro) diminuiu na sprint 4 (57% e 70%), enquanto o recall para a classe 2 se manteve praticamente o mesmo (70% e 67%). Considerando que o recall é a métrica mais importante, pode-se concluir que o desempenho dos modelos é bastante semelhante em ambas as sprints. Dessa forma, a escolha do melhor modelo dependerá da importância relativa das diferentes classes, a ser definida. 

### 10.4 Conclusão
&emsp;&emsp; Vale ressaltar que é importante considerar outros aspectos além do recall ao avaliar a adequação de um modelo para uma tarefa específica. A simplicidade e a eficiência computacional são fatores relevantes a serem considerados. Os modelos acima foram comparados entre si, pois eles trabalham de formas diferentes quando se tem base diferente. <br>







