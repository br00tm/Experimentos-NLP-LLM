import pandas as pd
import glob
import os
import re

# Configurações para exibir colunas largas
pd.set_option('display.max_colwidth', None)

# Dicionários expandidos e usando "raízes" das palavras
# Ex: "espera" pega "esperança", "esperar", "esperando"
dicionarios_raizes = {
    "Melancolica": [
        "trist", "dor", "sozin", "solid", "fim", "chuv", "noite", "frio", 
        "lagrima", "adeus", "medo", "vazio", "escuro", "perdi", "saudade", 
        "magoa", "choro", "sofr", "ausencia", "cinza", "morr", "morte"
    ],
    "Otimista": [
        "sol", "espera", "sonh", "luz", "amanha", "sorri", "novo", "vida", 
        "alegri", "venc", "amor", "brilh", "flor", "paz", "futuro", "acredit", 
        "voar", "livre", "festa", "felic", "bencao", "melhor"
    ],
    "Filosofica": [
        "tempo", "razao", "ser", "mundo", "limit", "porque", "destin", 
        "verdad", "exist", "pensar", "mente", "universo", "sentido", 
        "duvida", "questao", "saber", "human", "historia", "eterno", 
        "realidade", "ilusao"
    ]
}

def carregar_dados(pasta="dataset"):
    arquivos = glob.glob(os.path.join(pasta, "*.csv"))
    dfs = [pd.read_csv(arq) for arq in arquivos]
    # Garante que trata valores nulos como string vazia
    df_final = pd.concat(dfs, ignore_index=True)
    df_final['letra'] = df_final['letra'].fillna("")
    return df_final

def classificar_com_debug(letra):
    if not isinstance(letra, str): 
        return "Erro", {}, []
    
    letra_lower = letra.lower()
    scores = {cat: 0 for cat in dicionarios_raizes}
    palavras_encontradas = [] # Para debug
    
    for categoria, raizes in dicionarios_raizes.items():
        for raiz in raizes:
            # Usa Regex para contar ocorrências da raiz na letra
            # Ocorrencias de 'raiz'
            matches = len(re.findall(rf"{raiz}", letra_lower))
            if matches > 0:
                scores[categoria] += matches
                palavras_encontradas.append(f"{categoria}: {raiz} ({matches})")
    
    # Decide o vencedor
    # Se todos forem 0, define como Neutro
    if sum(scores.values()) == 0:
        vencedor = "Neutro/Sem palavras-chave"
    else:
        # Pega a categoria com maior valor
        vencedor = max(scores, key=scores.get)
    
    return vencedor, scores, palavras_encontradas

# --- Execução ---
df = carregar_dados()

# Aplica a função
resultado_completo = df['letra'].apply(classificar_com_debug)

# Separa os resultados em colunas
df['classificacao'] = resultado_completo.apply(lambda x: x[0])
df['pontuacao'] = resultado_completo.apply(lambda x: x[1])
df['palavras_achadas'] = resultado_completo.apply(lambda x: x[2])

# --- Relatório Visual ---
print("=== Resumo das Classificações ===")
print(df['classificacao'].value_counts())

print("\n=== Detalhe das Músicas (Amostra de 10) ===")
# Mostra Título, Classificação e QUAIS palavras definiram isso
cols_to_show = ['titulo', 'classificacao', 'palavras_achadas']
print(df[cols_to_show].head(10))

# Opcional: Salvar em arquivo para você ler com calma
df[cols_to_show].to_csv("resultado_classificacao_v2.csv", index=False)
print("\nArquivo 'resultado_classificacao_v2.csv' salvo para conferência detalhada.")