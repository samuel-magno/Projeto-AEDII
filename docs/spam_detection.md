# Detecção de Spam em SMS 

Esta ferramenta permite identificar spams em mensagens SMS a partir de um arquivo CSV de mensagens, exibindo métricas de acurácia e de performance.

---

## Como executar

```bash
python main.py \
  --data-path <caminho_para_spam_dataset.csv> \
  --classify \
  [--custom-patterns P1 P2 P3 ...] \
  [--vocab-size K] \
  [--threshold T] \
  [--search-reps R]
```

### Opções de linha de comando

- `--data-path`\
  Caminho para o seu arquivo CSV de SMS. Deve ter duas colunas: `label,text` (com cabeçalho, codificação Latin-1).

- `--classify`\
  Ativa o modo de detecção (classificação de SMS).

#### Seleção de padrões (um único grupo)

- `--custom-patterns P1 P2 ...`\
  Lista de palavras/frases fixas a serem usadas como "gatilhos" de spam.

  **ou**, se omitido:

- `--vocab-size K`\
  Extrai automaticamente os K termos mais discriminantes (via análise de frequência).

- `--threshold T` (default = `1`)\
  Número mínimo de padrões encontrados na mesma mensagem para classificá-la como spam.

- `--search-reps R` (default = `10`)\
  Quantas vezes repetir a classificação para medir tempo e estabilidade.

---

## O que acontece internamente

### 1. Carregamento

- SMS e rótulos são lidos de `--data-path`.

### 2. Seleção de padrões

- Se houver `--custom-patterns`, usa-se exatamente esses.
- Caso contrário, escolhe-se os top K (`--vocab-size`) padrões automáticos.

### 3. Classificação

- Para cada mensagem, é realizada uma contagem de quantos padrões aparecem (internamente via Boyer-Moore e Força Bruta)
- Se `count ≥ threshold`, marca **spam**; senão, **ham** (não spam).

### 4. Cálculo de métricas de acurácia

- **Matriz de Confusão** → TP, FP, TN, FN
- **Precision**, **Recall**, **F1-score**

### 5. Cálculo de métricas de performance

- Tempo total médio (e p90/p95) para classificar todo o arquivo
- Tempo médio por mensagem
- Desvio-padrão (stdev)
- Consumo aproximado de memória 

---

## Relatórios (`detection_reports/`)

- `details_<algoritmo>.csv`\
  Todas as mensagens, seu rótulo real, predição, contagem e lista de padrões encontrados.

- `spam_only_<algoritmo>.csv`\
  Mesmo, mas contendo apenas as linhas classificadas como spam.

---

## 🛠️ Exemplos de uso

### Exemplo 1: Padrões customizados

```bash
python main.py \
  --data-path spam_dataset.csv \
  --classify \
  --custom-patterns awarded guaranteed prize \
  --threshold 1 \
  --search-reps 3
```

> Usa “awarded”, “guaranteed” e “prize” e só marca spam se ≥1 aparecerem.

### Exemplo 2: Vocabulário automático

```bash
python main.py \
  --data-path spam_dataset.csv \
  --classify \
  --vocab-size 10 \
  --threshold 1 \
  --search-reps 5
```

> Extrai os 10 termos mais discriminantes do dataset e marca spam se qualquer um aparecer.

---

## 📊 Interpretação dos resultados

- **Matriz de Confusão** (TP, FP, TN, FN)\
  Avalia quantos spams reais foram identificados corretamente (TP), falsos positivos (FP) etc.

- **Precision** = TP / (TP + FP)\
  Fração de mensagens marcadas como spam que realmente eram spam.

- **Recall** = TP / (TP + FN)\
  Fração de spams reais capturados.

- **F1-score**\
  Balanço entre **precision** e **recall**.

- **Tempo médio total & por mensagem**\
  Mostra eficiência do algoritmo em seu dataset.

- **p90/p95**\
  Garante estabilidade: 90% das execuções ficaram abaixo desse tempo.

> Dessa forma, você consegue ajustar padrões, `threshold` e `vocab-size` para equilibrar precisão e cobertura da identificação de spams em mensagens, além de comparar o desempenho entre Boyer-Moore e Força Bruta nesse sentido