import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
from datetime import timedelta

# CONFIGURAÇÃO DA PÁGINA

st.set_page_config(
    page_title="Sistema de Alerta de Cheias",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Sistema de Previsão de Nível de Cheias")
st.markdown(f"Exibindo previsões de Nível para a estação de Jusante.")


# DEFINIÇÃO DE CAMINHOS

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA_PATH = os.path.join(BASE_PATH, "2_Dados_Processados")
OUTPUT_MODEL_PATH = os.path.join(BASE_PATH, "3_Modelos_Treinados")

CAMINHO_CSV_FINAL = os.path.join(OUTPUT_DATA_PATH, "dados_completos_limpos.csv")
CAMINHO_FEATURES = os.path.join(OUTPUT_MODEL_PATH, "features_lista_v1.joblib")
CAMINHO_RESULTADOS = os.path.join(OUTPUT_MODEL_PATH, "resultados_treinamento.json") 

DIAS_PREV_MAX = 5


# FUNÇÕES DE CARREGAMENTO E CÁLCULO 

@st.cache_data
def carregar_dados_historicos(caminho):
    """Carrega o CSV de dados limpos e calcula os níveis de alerta."""
    try:
        df_historico = pd.read_csv(caminho, index_col='data_hora', parse_dates=True)
        
        # CÁLCULO AUTOMÁTICO DAS COTAS DE ALERTA 
        cotas = {}
        cotas['atencao'] = df_historico['Nivel_J'].quantile(0.90)
        cotas['alerta'] = df_historico['Nivel_J'].quantile(0.95)
        cotas['inundacao'] = df_historico['Nivel_J'].quantile(0.98)
        
        return df_historico, cotas
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo de dados '{CAMINHO_CSV_FINAL}' não encontrado! "
                 f"Certifique-se que você executou o 'treinamento_offline.py' primeiro, "
                 f"e que as pastas '2_Dados_Processados' e '3_Modelos_Treinados' estão ao lado deste app.")
        return None, None
    except Exception as e:
        st.error(f"Erro ao ler os dados históricos: {e}")
        return None, None

@st.cache_resource
def carregar_modelos_ia(caminho_modelos_dir, caminho_features):
    """Carrega a lista de features e os 5 modelos (d1 a d5)."""
    modelos = {}
    try:
        features = joblib.load(caminho_features)
        
        for i in range(1, DIAS_PREV_MAX + 1):
            caminho_modelo_dia = os.path.join(caminho_modelos_dir, f"modelo_ia_cheias_d{i}.joblib")
            modelos[f'd{i}'] = joblib.load(caminho_modelo_dia)
            
        return modelos, features
    except FileNotFoundError:
        st.error(f"ERRO: Arquivos de modelo não encontrados! (ex: 'modelo_ia_cheias_d1.joblib') "
                 f"Execute o 'treinamento_offline.py' (versão com loop) primeiro.")
        return None, None
    except Exception as e:
        st.error(f"Erro ao carregar os modelos de IA: {e}")
        return None, None

@st.cache_data
def carregar_resultados_treinamento(caminho):
    """Carrega o JSON com os resultados (R2, MAE) do treinamento."""
    try:
        with open(caminho, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Arquivo de resultados '{CAMINHO_RESULTADOS}' não encontrado. "
                   "Execute o 'treinamento_offline.py' para ver as métricas de performance.")
        return None
    except Exception as e:
        st.error(f"Erro ao ler arquivo de resultados: {e}")
        return None

def criar_features_para_previsao(df):
    """Recria as mesmas features que o modelo foi treinado."""
    
    df_feat = df.copy()
    df_feat['dia_sin'] = np.sin(2 * np.pi * df_feat.index.dayofyear / 365.25)
    df_feat['dia_cos'] = np.cos(2 * np.pi * df_feat.index.dayofyear / 365.25)

    # Lags
    cols_para_lag = ['Nivel_M', 'Chuva_M', 'Nivel_J']
    for col in cols_para_lag:
        if col in df_feat.columns:
                 for lag in [1, 3, 5, 7]: 
                     df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)

    # Médias Móveis
    cols_para_media = ['Nivel_M', 'Vazao_M', 'Nivel_J']
    for col in cols_para_media:
        if col in df_feat.columns:
            df_feat[f'{col}_med_7d'] = df_feat[col].shift(1).rolling(7, min_periods=1).mean()
            df_feat[f'{col}_med_15d'] = df_feat[col].shift(1).rolling(15, min_periods=1).mean()

    # Somas Móveis (Chuva)
    if 'Chuva_M' in df_feat.columns:
        df_feat[f'Chuva_M_acum_3d'] = df_feat[f'Chuva_M'].shift(1).rolling(3, min_periods=1).sum()
        df_feat[f'Chuva_M_acum_7d'] = df_feat[f'Chuva_M'].shift(1).rolling(7, min_periods=1).sum()
        df_feat[f'Chuva_M_acum_15d'] = df_feat[f'Chuva_M'].shift(1).rolling(15, min_periods=1).sum()

    return df_feat

def get_status_cor(status_str):
    """Retorna uma cor com base no status de alerta."""
    if "INUNDAÇÃO" in status_str: return "red"
    if "ALERTA" in status_str: return "orange"
    if "ATENÇÃO" in status_str: return "yellow"
    return "green"

# CORPO PRINCIPAL DA INTERFACE

# Carregar Dados e Cotas de Alerta
df_historico, cotas = carregar_dados_historicos(CAMINHO_CSV_FINAL)
if df_historico is None: st.stop()

# Carregar Modelos de IA
modelos_ia, lista_de_features = carregar_modelos_ia(OUTPUT_MODEL_PATH, CAMINHO_FEATURES)
if modelos_ia is None: st.stop()

# Carregar Resultados do Treinamento
resultados_treinamento = carregar_resultados_treinamento(CAMINHO_RESULTADOS)

# Pega o último dia disponível nos dados como "Hoje"
ultimo_dia_dados = df_historico.index.max()
dados_atuais = df_historico.loc[ultimo_dia_dados]
dados_ontem = df_historico.loc[ultimo_dia_dados - timedelta(days=1)]

# LAYOUT COM ABAS
tab1, tab2, tab3 = st.tabs([
    "Painel de Previsão", 
    "Dados Históricos", 
    "ℹSobre o Sistema"
])


# PAINEL DE PREVISÃO 
with tab1:
    st.header(f"Condições em {ultimo_dia_dados.strftime('%d/%m/%Y')}")
    
    # Métricas de Condições Atuais
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Nível Atual (Jusante)", 
        f"{dados_atuais.get('Nivel_J', 0):.0f} cm",
        f"{dados_atuais.get('Nivel_J', 0) - dados_ontem.get('Nivel_J', 0):.0f} cm vs. ontem",
        delta_color="inverse"
    )
    col2.metric(
        "Nível Atual (Montante)", 
        f"{dados_atuais.get('Nivel_M', 0):.0f} cm",
        f"{dados_atuais.get('Nivel_M', 0) - dados_ontem.get('Nivel_M', 0):.0f} cm vs. ontem",
        delta_color="inverse"
    )
    chuva_recente = df_historico['Chuva_M'].tail(3).sum()
    col3.metric(
        "Chuva Acum. 3d (Montante)", 
        f"{chuva_recente:.1f} mm"
    )
    
    st.markdown("---")
    
    # Lógica de Previsão
    st.header(f"Previsão de Nível (Jusante) para os próximos {DIAS_PREV_MAX} dias")
    
    try:
        # Criar as features com base no histórico completo
        df_features_completo = criar_features_para_previsao(df_historico)
        
        # Selecionar a linha de features mais recente (do último dia)
        features_para_prever = df_features_completo.loc[ultimo_dia_dados][lista_de_features]

        # Fazer a previsão
        previsoes = []
        for i in range(1, DIAS_PREV_MAX + 1):
            modelo_dia = modelos_ia[f'd{i}']
            previsao_cm = modelo_dia.predict(features_para_prever.to_frame().T)[0]
            data_previsao = ultimo_dia_dados + timedelta(days=i)
            previsoes.append({
                "Data_Obj": data_previsao,
                "Data_Str": data_previsao.strftime('%d/%m/%Y'),
                "Nível": previsao_cm
            })
        
        # Criar DataFrame de Previsões
        df_previsoes = pd.DataFrame(previsoes)
        
        def aplicar_alerta(nivel):
            if nivel > cotas['inundacao']: return "🔴 INUNDAÇÃO"
            elif nivel > cotas['alerta']: return "🟠 ALERTA"
            elif nivel > cotas['atencao']: return "🟡 ATENÇÃO"
            else: return "🟢 ESTÁVEL"
        
        df_previsoes['Status'] = df_previsoes['Nível'].apply(aplicar_alerta)
        df_previsoes['Previsão (cm)'] = df_previsoes['Nível'].apply(lambda x: f"{x:.0f}")

        # Exibir Previsões como Métricas
        cols_previsao = st.columns(DIAS_PREV_MAX)
        for i, col in enumerate(cols_previsao):
            prev = df_previsoes.iloc[i]
            delta_vs_hoje = prev['Nível'] - dados_atuais.get('Nivel_J', 0)
            
            with col:
                st.metric(
                    label=f"D+{i+1} ({prev['Data_Str']})",
                    value=f"{prev['Nível']:.0f} cm",
                    delta=f"{delta_vs_hoje:.0f} cm vs. hoje",
                    delta_color="inverse"
                )
                cor = get_status_cor(prev['Status'])
                st.markdown(f"Status: **:{cor}[{prev['Status'].split(' ')[1]}]**")
        
        st.markdown("---")

        # Gráfico Principal
        st.subheader("Histórico Recente e Previsões Futuras")
        df_recente = df_historico.tail(90).copy()
        fig = go.Figure()

        # Nível Histórico
        fig.add_trace(go.Scatter(
            x=df_recente.index, y=df_recente['Nivel_J'],
            mode='lines', name='Nível Histórico (Jusante)',
            line=dict(color='blue')
        ))

        # Linhas de Alerta
        fig.add_hline(y=cotas['atencao'], line=dict(color='yellow', dash='dash'), name=f"Atenção ({cotas['atencao']:.0f} cm)")
        fig.add_hline(y=cotas['alerta'], line=dict(color='orange', dash='dash'), name=f"Alerta ({cotas['alerta']:.0f} cm)")
        fig.add_hline(y=cotas['inundacao'], line=dict(color='red', dash='dash'), name=f"Inundação ({cotas['inundacao']:.0f} cm)")

        # Pontos de Previsão
        fig.add_trace(go.Scatter(
            x=df_previsoes['Data_Obj'], 
            y=df_previsoes['Nível'], 
            mode='lines+markers',
            marker=dict(color='red', size=10, symbol='x'),
            line=dict(color='red', dash='dot'),
            name=f"Previsão Futura"
        ))

        fig.update_layout(
            title="Nível do Rio (Jusante) - Últimos 90 dias + Previsão",
            xaxis_title="Data",
            yaxis_title="Nível (cm)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de Previsões (em um Expander)
        with st.expander("Ver Tabela Detalhada da Previsão"):
            st.dataframe(
                df_previsoes[['Data_Str', 'Previsão (cm)', 'Status']],
                use_container_width=True,
                hide_index=True,
                column_config={"Data_Str": "Data"}
            )

    except Exception as e:
        st.error(f"Ocorreu um erro durante a criação das features ou na previsão: {e}")
        st.exception(e)


# DADOS HISTÓRICOS
with tab2:
    st.header("Análise dos Dados Históricos")
    st.markdown("Arraste nos gráficos para dar zoom e clique duas vezes para resetar.")
    
    df_hist_plot = df_historico.tail(365 * 2) # Pegar últimos 2 anos para plotar

    # Gráfico de Níveis
    st.subheader("Histórico de Níveis (Montante vs. Jusante)")
    fig_niveis = go.Figure()
    fig_niveis.add_trace(go.Scatter(
        x=df_hist_plot.index, y=df_hist_plot['Nivel_J'], 
        name='Nível Jusante', line=dict(color='blue')
    ))
    if 'Nivel_M' in df_hist_plot.columns:
        fig_niveis.add_trace(go.Scatter(
            x=df_hist_plot.index, y=df_hist_plot['Nivel_M'], 
            name='Nível Montante', line=dict(color='lightblue', dash='dot')
        ))
    fig_niveis.update_layout(yaxis_title="Nível (cm)", hovermode="x unified")
    st.plotly_chart(fig_niveis, use_container_width=True)

    # Gráfico de Chuva
    st.subheader("Histórico de Chuva (Montante)")
    if 'Chuva_M' in df_hist_plot.columns:
        fig_chuva = go.Figure()
        fig_chuva.add_trace(go.Bar(
            x=df_hist_plot.index, y=df_hist_plot['Chuva_M'], 
            name='Chuva (mm)'
        ))
        fig_chuva.update_layout(yaxis_title="Chuva (mm)", hovermode="x unified")
        st.plotly_chart(fig_chuva, use_container_width=True)
    else:
        st.info("Não foram encontrados dados de Chuva (Chuva_M) para plotar.")

    # Tabela de Dados Recentes
    st.subheader("Últimos 10 Registros Históricos")
    st.dataframe(df_historico.tail(10).sort_index(ascending=False), use_container_width=True)


# SOBRE O SISTEMA
with tab3:
    st.header("Sobre o Sistema")
    st.markdown("""
    Este sistema utiliza modelos de **Inteligência Artificial (XGBoost)** para prever o nível
    do rio na estação de Jusante com base nos dados históricos das estações de Montante e Jusante.
    
    Cinco modelos separados foram treinados, um para cada dia de previsão (D+1 a D+5).
    """)
    
    # Métricas de Performance
    st.subheader("Desempenho dos Modelos (Aferido em Dados de Teste)")
    if resultados_treinamento:
        df_resultados = pd.DataFrame.from_dict(resultados_treinamento, orient='index')
        df_resultados['MAE_cm'] = df_resultados['MAE_cm'].map('{:.2f} cm'.format)
        df_resultados['R2 (Acurácia)'] = df_resultados['R2'].map('{:.2%}'.format)
        
        st.table(df_resultados[['R2 (Acurácia)', 'MAE_cm']])
    else:
        st.markdown("Resultados de performance não encontrados. (Execute `treinamento_offline.py`.)")

    # Limiares de Risco
    st.subheader("Limiares de Risco (Calculados Automaticamente)")
    st.markdown("""
    Os limiares de risco são definidos automaticamente usando **percentis** dos dados históricos
    de Nível da estação de Jusante.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Cota de Atenção (Percentil 90)", f"{cotas['atencao']:.0f} cm")
    col2.metric("Cota de Alerta (Percentil 95)", f"{cotas['alerta']:.0f} cm")
    col3.metric("Cota de Inundação (Percentil 98)", f"{cotas['inundacao']:.0f} cm")