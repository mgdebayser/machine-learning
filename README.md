# machine-learning

Predição de turnos de interação em diálogos conversacionais utilizando os seguintes algoritmos de aprendizagem de máquina:

- MLE: Maximum Likelihood Estimation = Máxima Verossimilhança
https://pt.wikipedia.org/wiki/M%C3%A1xima_verossimilhan%C3%A7a

- K-Means Clustering = Agrupamento K-means
https://pt.wikipedia.org/wiki/K-means

- LDA: Latent Dirichlet allocation = Alocação latente de Dirichlet
https://pt.wikipedia.org/wiki/Aloca%C3%A7%C3%A3o_latente_de_Dirichlet

- Word Embedding with Gensim
https://radimrehurek.com/gensim/models/word2vec.html

## Arquitetura MLE
Artigo: https://arxiv.org/pdf/1907.02090.pdf

![Arquitetura utilizando MLE](/img/mle-kmeans-dialogues.png)

Este repositório contém diálogos de entre usuários humanos, um chatbot mediador e chatbots de recomendação de opções de investimento (poupança, tesouro direto e CDB). O sistema está descrito no artigo.

Para treinar os modelos de MLE, utilize o seguinte script:
```
python mle_word2vec_kmeans_finch_dialogues.py
```