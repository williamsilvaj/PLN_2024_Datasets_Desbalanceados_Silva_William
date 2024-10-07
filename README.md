# Atividade 3 – Uso de Datasets Desbalanceados

**Autor:** William Santos Silva  
**Data:** Outubro 2024

## Introdução
Nesta atividade, abordaremos o uso de datasets desbalanceados em tarefas de classificação, utilizando um dataset específico contendo um mínimo de 10.000 registros. O objetivo é explorar as etapas de análise e pré-processamento necessárias para preparar os dados para o treinamento de modelos de aprendizado de máquina.

## Dataset Selecionado
O dataset selecionado para esta atividade é intitulado **Emails for Spam or Ham Classification** e pode ser acessado através da seguinte URL: [Dataset URL](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007).

### a) Campos que Compoem Cada Registro
Os campos que compõem cada registro do dataset são:
- **label:** Indica se o e-mail é spam ou não.
- **text:** Contém o conteúdo do e-mail.

### b) Labels das Classes Existentes
Os labels das classes existentes no dataset são:
- **1:** spam
- **0:** não spam

### c) Quantidade de Registros por Label
A quantidade de registros que não contenham label, que contenham um label, dois labels, até o máximo de labels existentes é apresentada na figura abaixo:

![Registro com um e zero etiquetas](c.png)

### d) Quantidade de Registros Associados a Cada Label/Classe Existente
A quantidade de registros associados a cada label/classe existente está mostrada na figura abaixo:

![Quantidade de label por classe](d.png)

### e) Registros com e Sem Labels/Classes
A quantidade de registros com labels/classes e sem labels/classes pode ser visualizada na figura abaixo:

![Quantidade de registros com e sem labels](e.png)

### Amostragem do Dataset
Para garantir um equilíbrio controlado entre as classes spam e não spam, realizamos uma amostragem do dataset original. O objetivo foi extrair uma amostra desbalanceada composta por 10.000 registros, onde 90% dos registros pertencem à classe "não spam" (ham) e 10% à classe "spam".

A amostragem foi realizada da seguinte forma:

1. **Separação das Classes:** Os registros do dataset foram inicialmente separados nas classes spam e não spam (ham) com base no valor do campo `label`.
    ```python
    spam_dataframe = dataframe[dataframe['label'] == 1]
    ham_dataframe = dataframe[dataframe['label'] == 0]
    ```

2. **Definição da Contagem:** Para a amostra final, estabelecemos que deveriam ser incluídos 9.000 registros da classe "não spam" e 1.000 registros da classe "spam".
    ```python
    ham_count = 9000  # 90% de 10,000
    spam_count = 1000  # 10% de 10,000
    ```

3. **Seleção das Amostras:** Em seguida, foram selecionados aleatoriamente 9.000 registros da classe "não spam" e 1.000 registros da classe "spam".
    ```python
    ham_sample = ham_dataframe.sample(n=ham_count, random_state=42)
    spam_sample = spam_dataframe.sample(n=spam_count, random_state=42)
    ```

A amostragem resultou em um conjunto de dados que contém 9.000 registros da classe "não spam" e 1.000 registros da classe "spam".

![Amostra: 10% spam e 90% não spam](h.png)

### Tratamento de Dados
Nesta etapa, foram realizadas algumas operações importantes para o pré-processamento do dataset:

1. **Verificação de Registros Duplicados:** Inicialmente, foi feita a identificação de registros duplicados no DataFrame.
   
2. **Criação de Coluna de Labels:** Não foi necessário criar uma nova coluna para indicar a presença de labels, pois todos os registros já possuíam um label associado.

3. **Concatenção de Campos Textuais:** Após análise, verificou-se que não havia a necessidade de concatenar campos textuais.

Essas operações asseguram que o dataset esteja limpo e pronto para as próximas etapas de análise e modelagem.

### Distribuição de Palavras por Registro
A distribuição da quantidade de palavras por registro está ilustrada na figura abaixo:

![Quantidade de palavras por registro](p.png)

### Criação dos Conjuntos de Treinamento, Validação e Teste
Utilizando a biblioteca Scikit-multilearn, foram criados os conjuntos de registros para treinamento, validação e teste.
- Tamanho do conjunto de treinamento: 2532
- Tamanho do conjunto de validação: 1265
- Tamanho do conjunto de teste: 1267
- Tamanho do conjunto sem labels: 4936

### Training Slices
Os training slices do dataset foram criados conforme segue:
- Tamanhos dos splits de target: [8, 16, 32, 64, 128, 2532]
- Tamanhos reais dos splits: [8, 16, 32, 64, 128, 2532]

### Implementação do Naive Bayesline
A implementação do Naive Bayesline foi realizada, e os gráficos para Micro e Macro F1 score estão apresentados na figura abaixo.

O Naive Bayesline é uma técnica de classificação que se baseia no teorema de Bayes e faz a suposição de independência entre as características. 

Os gráficos apresentados mostram as pontuações de Micro F1 e Macro F1, que são duas métricas usadas para avaliar a eficácia do classificador:

- **Micro F1 Score:** Considera o número total de verdadeiros positivos, falsos negativos e falsos positivos em todas as classes e calcula uma média ponderada.
  
- **Macro F1 Score:** Calcula o F1 Score para cada classe individualmente e, em seguida, calcula a média desses valores.

![Macro F1 e Micro F1 scores](jk.png)

### Uso de Embeddings e Resultados
Utilizando o recurso de embeddings e o modelo GPT-2 disponível no Hugging Face, juntamente com os recursos oferecidos pelo FAISS, foram apresentados e discutidos os resultados de micro e macro F1 score, conforme mostrado na figura abaixo.

Os gráficos demonstram a eficácia da abordagem de embeddings em comparação com a implementação do Naive Bayesline.

- **Micro F1 Score:** A pontuação Micro F1 apresenta um aumento consistente à medida que o número de amostras de treinamento aumenta.

- **Macro F1 Score:** A pontuação Macro F1 também mostra uma tendência de crescimento, especialmente em amostras maiores.

![Micro e Macro scores](fr.png)

![Micro F1 e Macro F1 scores](sdsd.png)

## Conclusão
A implementação dos embeddings a partir do modelo GPT-2 não só melhorou os resultados das pontuações de F1, mas também trouxe à tona novas perspectivas sobre como os dados podem ser manipulados e representados para maximizar a eficácia da classificação. Esta análise fornece uma base sólida para futuras investigações e possíveis aprimoramentos no modelo.
