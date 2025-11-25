import pandas as pd
import glob
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÕES ---
ARQUIVO_SAIDA = "resultado_similaridade_detalhado.json"
MODELO_NOME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MINIMO_PARA_DETALHAR = 0.50 # Só busca trechos se a música for pelo menos 50% similar

def carregar_dados(pasta="dataset"):
    arquivos = glob.glob(os.path.join(pasta, "*.csv"))
    dfs = []
    for arq in arquivos:
        try:
            dfs.append(pd.read_csv(arq))
        except:
            pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=['letra']).reset_index(drop=True)

def encontrar_melhores_trechos(letra_a, letra_b, model, top_k=3):
    """
    Quebra as letras em linhas e encontra os versos que mais se parecem semanticamente.
    """
    # Quebra em linhas e remove vazias
    versos_a = [v.strip() for v in letra_a.split('\n') if v.strip()]
    versos_b = [v.strip() for v in letra_b.split('\n') if v.strip()]
    
    if not versos_a or not versos_b:
        return []

    # Gera embeddings para cada verso individualmente
    emb_a = model.encode(versos_a)
    emb_b = model.encode(versos_b)
    
    # Matriz de similaridade entre versos (Linhas A vs Linhas B)
    matriz = cosine_similarity(emb_a, emb_b)
    
    pares_versos = []
    
    # Itera sobre a matriz para pegar todas as combinações
    for i in range(len(versos_a)):
        for j in range(len(versos_b)):
            score = float(matriz[i][j])
            if score > 0.6: # Só guarda se o verso for relevante
                pares_versos.append({
                    "verso_musica_1": versos_a[i],
                    "verso_musica_2": versos_b[j],
                    "score_verso": round(score, 4)
                })
    
    # Ordena pelos versos mais parecidos e pega os Top K
    pares_versos.sort(key=lambda x: x['score_verso'], reverse=True)
    return pares_versos[:top_k]

def gerar_analise_detalhada():
    df = carregar_dados()
    if df.empty: return

    print(f"🧠 Carregando modelo e processando {len(df)} músicas...")
    model = SentenceTransformer(MODELO_NOME)
    
    # 1. Embeddings Gerais (Música inteira)
    print("   -> Calculando similaridade global...")
    embeddings_gerais = model.encode(df['letra'].tolist(), show_progress_bar=True)
    matriz_global = cosine_similarity(embeddings_gerais)
    
    lista_resultados = []
    num_musicas = len(df)

    print("🔎 Comparando versos nas músicas similares (isso pode demorar um pouco)...")
    
    for i in range(num_musicas):
        for j in range(i + 1, num_musicas):
            score_global = float(matriz_global[i][j])
            
            # Otimização: Só detalha trechos se a similaridade geral for relevante
            if score_global >= MINIMO_PARA_DETALHAR:
                
                print(f"   ... Analisando trechos: {df.iloc[i]['titulo']} x {df.iloc[j]['titulo']}")
                
                trechos = encontrar_melhores_trechos(
                    df.iloc[i]['letra'], 
                    df.iloc[j]['letra'], 
                    model
                )
                
                obj = {
                    "musica_1": df.iloc[i]['titulo'],
                    "musica_2": df.iloc[j]['titulo'],
                    "similaridade_global": round(score_global, 4),
                    "trechos_similares": trechos
                }
                lista_resultados.append(obj)

    # Ordena pelo score global
    lista_resultados.sort(key=lambda x: x["similaridade_global"], reverse=True)

    print(f"💾 Salvando {ARQUIVO_SAIDA}...")
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(lista_resultados, f, indent=4, ensure_ascii=False)

    print("✅ Concluído.")

if __name__ == "__main__":
    gerar_analise_detalhada()