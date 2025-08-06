import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Algoritmos (como definidos antes)
def brute_force(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            return i
    return -1

def build_bad_char_table(pattern):
    table = [-1] * 256
    for i in range(len(pattern)):
        table[ord(pattern[i])] = i
    return table

def boyer_moore(text, pattern):
    n, m = len(text), len(pattern)
    bad_char = build_bad_char_table(pattern)
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            return s
        s += max(1, j - bad_char[ord(text[s + j])] if ord(text[s + j]) < 256 else 1)
    return -1

# Carregando o CSV
df = pd.read_csv("spam_dataset.csv", encoding="latin1")
spam_messages = df[df['v1'] == 'spam']['v2'].head(400)

# Padrão de busca
pattern = "limited time offer"
print(f"pattern: {pattern}\n")

# Avaliação dos algoritmos
print("Força Bruta:")
start = time.time()
for msg in spam_messages:
    brute_force(msg.lower(), pattern.lower())
end = time.time()
tempo_bruta = end - start
print(f"Tempo total: {tempo_bruta:.5f} s\n")

print("Boyer-Moore:")
start = time.time()
for msg in spam_messages:
    boyer_moore(msg.lower(), pattern.lower())
end = time.time()
tempo_boyer = end - start
print(f"Tempo total: {tempo_boyer:.5f} s")

# Resultados em gráficos
tempos = {
     'Força Bruta': tempo_bruta,
     'Boyer-Moore': tempo_boyer
}

algorithms = list(tempos.keys())
execution_times = list(tempos.values())

plt.figure(figsize=(8, 5))
plt.bar(algorithms, execution_times, color=['blue', 'green'])
plt.ylabel('Tempo de Execução (segundos)')
plt.title('Comparação do Tempo de Execução dos Algoritmos de Busca de Padrão')

plt.show()