import pandas as pd
import numpy as np
import os
import joblib
import warnings
import glob 
import re 
import json 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor 

warnings.filterwarnings('ignore')

# CONFIGURAÇÃO DE CAMINHOS

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_PATH, "1_Dados_Brutos") 
OUTPUT_DATA_PATH = os.path.join(BASE_PATH, "2_Dados_Processados")
OUTPUT_MODEL_PATH = os.path.join(BASE_PATH, "3_Modelos_Treinados")

# CÓDIGOS DAS ESTAÇÕES
COD_MONTANTE_FLU = "56991500" 
COD_MONTANTE_PLU = "1941008" 
COD_JUSANTE = "56992000"

# CAMINHOS DE SAÍDA
CAMINHO_CSV_FINAL = os.path.join(OUTPUT_DATA_PATH, "dados_completos_limpos.csv")
CAMINHO_FEATURES = os.path.join(OUTPUT_MODEL_PATH, "features_lista_v1.joblib")
CAMINHO_RESULTADOS = os.path.join(OUTPUT_MODEL_PATH, "resultados_treinamento.json") # <--- ADICIONADO

os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

print(f">>> Base do Projeto: {BASE_PATH}")

# PROCESSAMENTO DE DADOS

def carregar_dados_hidroweb(codigo_estacao, pasta_dados):
    """Lê e processa os arquivos CSV de formato "largo" (wide) do HidroWeb."""
    print(f" -> Processando estação: {codigo_estacao}")
    padrao_busca = os.path.join(pasta_dados, f"*{codigo_estacao}*.csv")
    arquivos_encontrados = glob.glob(padrao_busca)
    
    if not arquivos_encontrados:
        print(f"   - ERRO: Nenhum arquivo CSV encontrado para {codigo_estacao} em '{pasta_dados}'")
        return pd.DataFrame()

    lista_dfs_tipo = [] 
    for caminho_arquivo in arquivos_encontrados:
        nome_arquivo = os.path.basename(caminho_arquivo).lower()
        tipo_dado = None
        colunas_prefixo = None
        
        if "cota" in nome_arquivo:
            tipo_dado = "nivel"; colunas_prefixo = "Cota"
        elif "vazao" in nome_arquivo or "vazões" in nome_arquivo or "vazoes" in nome_arquivo:
            tipo_dado = "vazao"; colunas_prefixo = "Vazao"
        elif "chuva" in nome_arquivo:
            tipo_dado = "chuva"; colunas_prefixo = "Chuva"
        
        if tipo_dado is None:
            print(f"   - Aviso: Ignorando arquivo (tipo não reconhecido): {nome_arquivo}")
            continue

        print(f"   - Lendo {tipo_dado} (formato WIDE) de: {nome_arquivo}")
        try:
            LINHAS_A_PULAR = 14
            try:
                with open(caminho_arquivo, 'r', encoding='latin-1') as f:
                    for i, line in enumerate(f):
                        if 'EstacaoCodigo' in line and ('NivelConsistencia' in line or 'Data' in line):
                            LINHAS_A_PULAR = i; break
                        if i > 20: break
            except Exception as e:
                print(f"     - (Aviso) Não foi possível detectar linhas de cabeçalho, usando padrão {LINHAS_A_PULAR}. Erro: {e}")
            
            df_wide = pd.read_csv(
                caminho_arquivo, sep=';', decimal=',',
                skiprows=LINHAS_A_PULAR, encoding='latin-1',
                engine='python', on_bad_lines='skip'
            )

            if 'MediaDiaria' in df_wide.columns:
                df_wide_diario = df_wide[df_wide['MediaDiaria'] == 1].copy()
            else:
                print(f"     - (Info) Coluna 'MediaDiaria' não encontrada. Assumindo dados diários (ex: Chuva).")
                df_wide_diario = df_wide.copy()
            
            if df_wide_diario.empty:
                print(f"     - Aviso: Nenhum dado de MÉDIA DIÁRIA (MediaDiaria=1) encontrado em {nome_arquivo}.")
                continue

            colunas_dados = [c for c in df_wide_diario.columns if c.startswith(colunas_prefixo) and re.search(r'\d+$', c)]
            
            if not colunas_dados:
                print(f"     - ERRO: Nenhuma coluna de dados (ex: {colunas_prefixo}01) encontrada.")
                continue
            
            df_long = df_wide_diario.melt(
                id_vars=['Data'], value_vars=colunas_dados,
                var_name='Dia_Str', value_name=tipo_dado
            )
            
            df_long['Dia'] = df_long['Dia_Str'].str.extract(r'(\d+)$').astype(int)
            data_mes_ano = pd.to_datetime(df_long['Data'], format='%d/%m/%Y', errors='coerce')
            df_long['Mes'] = data_mes_ano.dt.month
            df_long['Ano'] = data_mes_ano.dt.year
            df_long['data_hora'] = pd.to_datetime(
                dict(year=df_long['Ano'], month=df_long['Mes'], day=df_long['Dia']),
                errors='coerce' 
            )
            
            df_long = df_long.dropna(subset=['data_hora', tipo_dado])
            df_long[tipo_dado] = pd.to_numeric(df_long[tipo_dado], errors='coerce')
            
            if tipo_dado == 'nivel':
               print(f"     - (Info) Dados de Nível lidos e assumidos como CM.")
            
            df_tipo_diario = df_long.set_index('data_hora')[[tipo_dado]]
            df_tipo_diario = df_tipo_diario[~df_tipo_diario.index.duplicated(keep='first')]
            df_tipo_diario = df_tipo_diario.sort_index()
            lista_dfs_tipo.append(df_tipo_diario)
            
        except Exception as e:
            print(f"     - ERRO ao processar o arquivo {nome_arquivo}: {e}")

    if not lista_dfs_tipo:
        print(f"   - ERRO: Nenhum dado (Cota, Vazão, Chuva) lido com sucesso para {codigo_estacao}.")
        return pd.DataFrame()

    df_estacao = pd.concat(lista_dfs_tipo, axis=1)
    df_estacao = df_estacao[~df_estacao.index.duplicated(keep='first')]
    return df_estacao

# LÓGICA DE CARREGAMENTO DAS 3 ESTAÇÕES
print("\n>>> [ETAPA 1] Carregamento e Limpeza dos CSVs Locais (Formato HidroWeb)")

print(f" -> Carregando Nível/Vazão de MONTANTE: {COD_MONTANTE_FLU}")
df_m_flu_bruto = carregar_dados_hidroweb(COD_MONTANTE_FLU, INPUT_PATH)

print(f" -> Carregando Chuva de MONTANTE: {COD_MONTANTE_PLU}")
df_m_plu_bruto = carregar_dados_hidroweb(COD_MONTANTE_PLU, INPUT_PATH)

print(f" -> Carregando Nível/Vazão de JUSANTE: {COD_JUSANTE}")
df_j_bruto = carregar_dados_hidroweb(COD_JUSANTE, INPUT_PATH)

if df_m_flu_bruto.empty or df_j_bruto.empty:
    print(f"\n!!! FALHA CRÍTICA: Verifique se os arquivos de Nivel/Vazao")
    print(f"!!! estão na pasta '{INPUT_PATH}' para os códigos {COD_MONTANTE_FLU} e {COD_JUSANTE}.")
    exit()

# Renomear e selecionar colunas
df_m_flu = df_m_flu_bruto.rename(columns={'nivel': 'Nivel_M', 'vazao': 'Vazao_M'})[['Nivel_M', 'Vazao_M']]
df_m_plu = df_m_plu_bruto.rename(columns={'chuva': 'Chuva_M'})[['Chuva_M']]
df_j = df_j_bruto.rename(columns={'nivel': 'Nivel_J', 'vazao': 'Vazao_J'})[['Nivel_J', 'Vazao_J']]

print(" -> Unindo dados de Montante (Fluvio + Pluvio)...")
df_m = df_m_flu.join(df_m_plu, how='outer')

print(" -> Unindo dados de Montante e Jusante...")
df_final = df_m.join(df_j, how='outer')

print(" -> Reamostrando, interpolando e preenchendo falhas...")
df_final = df_final.resample('D').mean()
df_final = df_final.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')

# Garantir que colunas existam
for col in ['Nivel_M', 'Vazao_M', 'Chuva_M', 'Nivel_J', 'Vazao_J']:
    if col not in df_final.columns or df_final[col].isnull().all():
        print(f" -> Aviso: Coluna '{col}' não encontrada ou vazia. Preenchendo com 0.0.")
        df_final[col] = 0.0
df_final = df_final.fillna(0)


# Remover Zeros Falsos do Final

print(" -> Removendo dados falsos (zeros) do final do arquivo...")
try:
    ultima_data_real = df_final[(df_final['Nivel_J'] > 1) | (df_final['Nivel_M'] > 1)].index.max()
    if pd.isna(ultima_data_real):
        print("!!! ERRO: Não foi encontrado nenhum dado de nível real (maior que 1cm).")
        exit()
    print(f" -> Último dia com dados reais encontrado: {ultima_data_real.date()}")
    df_final = df_final.loc[:ultima_data_real]
except Exception as e:
    print(f" -> ERRO ao tentar limpar os zeros do final: {e}"); exit()

df_final.to_csv(CAMINHO_CSV_FINAL, index_label='data_hora')
print(f" -> Planilha limpa salva: {len(df_final)} dias de registros reais.")
print(" -> A IA será treinada com estes dados.")

# TREINAMENTO IA

print("\n>>> [ETAPA 2] Treinamento IA para 5 dias de previsão")

if len(df_final) < 50:
    print("!!! ERRO FATAL: Menos de 50 dias de dados comuns encontrados.")
    exit()

# CRIAÇÃO DE FEATURES
print(" -> Criando features (com lógica de acúmulo de chuva)...")
df_train = df_final.copy()
cols_input = ['Nivel_M', 'Vazao_M', 'Chuva_M', 'Nivel_J']

df_train['dia_sin'] = np.sin(2 * np.pi * df_train.index.dayofyear / 365.25)
df_train['dia_cos'] = np.cos(2 * np.pi * df_train.index.dayofyear / 365.25)

# Lags
cols_para_lag = ['Nivel_M', 'Chuva_M', 'Nivel_J']
print(" -> Criando features de lag (deslocamento temporal)...")
for col in cols_para_lag:
    if col in df_train.columns and df_train[col].sum() != 0:
         print(f"     - Criando lags para: {col}")
         for lag in [1, 3, 5, 7]: 
             df_train[f'{col}_lag_{lag}'] = df_train[col].shift(lag)
    else:
         print(f"     - Aviso: Coluna {col} está vazia ou é 0. Lags não serão criados.")

# Médias Móveis
print(" -> Criando features de média móvel (tendência do rio)...")
cols_para_media = ['Nivel_M', 'Vazao_M', 'Nivel_J']
for col in cols_para_media:
    if col in df_train.columns and df_train[col].sum() != 0:
        print(f"     - Criando médias para: {col}")
        df_train[f'{col}_med_7d'] = df_train[col].shift(1).rolling(7, min_periods=1).mean()
        df_train[f'{col}_med_15d'] = df_train[col].shift(1).rolling(15, min_periods=1).mean()
    else:
         print(f"     - Aviso: Coluna {col} está vazia ou é 0. Médias não serão criadas.")

# Somas Móveis (Chuva)
print(" -> Criando features de chuva ACUMULADA (soma)...")
if 'Chuva_M' in df_train.columns and df_train['Chuva_M'].sum() != 0:
    print(f"     - Criando acúmulo para: Chuva_M")
    df_train[f'Chuva_M_acum_3d'] = df_train['Chuva_M'].shift(1).rolling(3, min_periods=1).sum()
    df_train[f'Chuva_M_acum_7d'] = df_train['Chuva_M'].shift(1).rolling(7, min_periods=1).sum()
    df_train[f'Chuva_M_acum_15d'] = df_train['Chuva_M'].shift(1).rolling(15, min_periods=1).sum()
else:
    print(f"     - Aviso: Coluna Chuva_M está vazia ou é 0. Acúmulo não será criado.")

# Salva a lista de features (colunas) que os modelos usarão
# (Remove colunas originais e o alvo)
feats = [c for c in df_train.columns if c not in cols_input]
joblib.dump(feats, CAMINHO_FEATURES)
print(f"\n>>> Lista de Features salva em '{CAMINHO_FEATURES}'")

# LOOP DE TREINAMENTO
print("\n>>> INICIANDO TREINAMENTO DOS 5 MODELOS <<<")
resultados = {}

for dias_previsao in range(1, 6):
    
    print(f"\n{'='*60}")
    print(f">>> Treinando Modelo para D+{dias_previsao} (Previsão de {dias_previsao} dia(s))")
    print(f"{'='*60}")
    
    # Criar o TARGET específico para este loop
    TARGET = f'Alvo_Nivel_J_d{dias_previsao}'
    df_train[TARGET] = df_final['Nivel_J'].shift(-dias_previsao)
    
    # Limpar NaNs (criados pelos shifts e pelo alvo)
    # Remove linhas onde o alvo OU as features são NaN
    df_loop = df_train.dropna(subset=[TARGET] + feats)
    
    if len(df_loop) < 30:
         print(f"!!! ERRO (D+{dias_previsao}): Após criar features, sobraram menos de 30 dias. Impossível treinar.")
         continue

    # Separar X e y para este loop
    X = df_loop[feats]
    y = df_loop[TARGET]

    # Separar em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f" -> D+{dias_previsao}: Treinando com {len(X_train)} dias, testando com {len(X_test)} dias.")
    print(f" -> Período de Treino: {X_train.index.min().date()} a {X_train.index.max().date()}")
    print(f" -> Período de Teste: {X_test.index.min().date()} a {X_test.index.max().date()}")

    # Definir e Treinar o Modelo
    print(" -> Treinando modelo XGBoost (Modo Otimizado)...")
    modelo = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False # Desliga o log por árvore para não poluir
    )
    
    # Avaliar e Salvar Resultados
    preds = modelo.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\n--- RESULTADOS (D+{dias_previsao}) ---")
    print(f"Acurácia (R²) no Teste: {r2:.4f}")
    print(f"Erro Médio Absoluto:     {mae:.2f} cm") 
    
    # Adiciona ao dicionário de resultados
    resultados[f"D+{dias_previsao}"] = {"R2": r2, "MAE_cm": mae}

    # Salvar o modelo específico deste dia
    caminho_modelo_dia = os.path.join(OUTPUT_MODEL_PATH, f"modelo_ia_cheias_d{dias_previsao}.joblib")
    joblib.dump(modelo, caminho_modelo_dia)
    print(f" -> Modelo D+{dias_previsao} salvo em '{caminho_modelo_dia}'")

print(f"\n{'-'*60}")
print(">>> TREINAMENTO CONCLUÍDO <<<")
print(f">>> Planilha limpa salva em '{OUTPUT_DATA_PATH}'")
print("\n--- RESUMO DOS RESULTADOS ---")

for dia, metricas in resultados.items():
    print(f"Modelo {dia}: Acurácia (R²) = {metricas['R2']:.4f}, Erro Médio = {metricas['MAE_cm']:.2f} cm")

# BLOCO ADICIONADO PARA SALVAR RESULTADOS
print(f"\n>>> Salvando resultados em {CAMINHO_RESULTADOS}...")
with open(CAMINHO_RESULTADOS, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=4)
    
print(f"\n>>> SUCESSO! 5 modelos salvos em '{OUTPUT_MODEL_PATH}'")
print("\n>>> Agora você pode rodar o 'streamlit run app.py'.")