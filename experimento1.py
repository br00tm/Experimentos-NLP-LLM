import pandas as pd
import glob
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Configurações iniciais
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')
# Adicionando stopwords manuais citadas no PDF
stop_words.extend(["e", "de", "para", "que", "o", "a", "um", "uma", "pra"]) 

def carregar_dados(pasta="dataset"):
    arquivos = glob.glob(os.path.join(pasta, "*.csv"))
    dfs = []
    for arq in arquivos:
        try:
            dfs.append(pd.read_csv(arq))
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
    if not dfs:
        raise ValueError("Nenhum arquivo CSV encontrado na pasta dataset.")
    return pd.concat(dfs, ignore_index=True)

def limpar_texto(texto):
    # Passo 1: Limpeza e lowercase [cite: 8, 9]
    if not isinstance(texto, str): return ""
    palavras = texto.lower().split()
    palavras_limpas = [p for p in palavras if p.isalpha() and p not in stop_words]
    return " ".join(palavras_limpas)

# Execução
df = carregar_dados()
# Ajuste o nome da coluna 'letra' se necessário
df['letra_limpa'] = df['letra'].apply(limpar_texto)

# --- 1. Contagem Simples [cite: 10] ---
todas_palavras = " ".join(df['letra_limpa']).split()
contagem = Counter(todas_palavras)
print("=== Top 20 Palavras Mais Frequentes (Vocabulário Geral) ===") # [cite: 11]
print(contagem.most_common(20))

# --- 2. TF-IDF [cite: 12] ---
tfidf = TfidfVectorizer()
matriz_tfidf = tfidf.fit_transform(df['letra_limpa'])
feature_names = tfidf.get_feature_names_out()

print("\n=== Top Termos TF-IDF por Música (Exemplo: 15 músicas) ===")
for i in range(min(15, len(df))):
    feature_index = matriz_tfidf[i, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [matriz_tfidf[i, x] for x in feature_index])
    # Ordena por score para achar a palavra mais importante daquela música [cite: 13, 14]
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    top_termos = [feature_names[i] for i, s in sorted_scores[:5]] # [cite: 15]
    print(f"Música: {df.iloc[i].get('titulo', 'Desconhecida')} -> Termos chave: {top_termos}")

# --- 3. Nuvem de Palavras [cite: 16] ---
# Usa a frequência da contagem simples para gerar o visual
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(contagem)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Nuvem de Palavras - Temas Principais")
plt.show()