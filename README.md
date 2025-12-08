# 🌊 Sistema de Alerta de Cheias & Previsão Hidrológica com IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

Este projeto é uma solução de **Ciência de Dados** ponta a ponta (End-to-End) para monitoramento e previsão de níveis de rios. O sistema processa dados brutos de estações telemétricas, treina modelos de Machine Learning e disponibiliza um dashboard interativo para auxílio na tomada de decisões sobre riscos de inundação.

O modelo é capaz de prever o nível do rio na estação de Jusante com até **5 dias de antecedência (D+1 a D+5)**.

---

## 🏗️ Arquitetura do Projeto

O sistema é dividido em dois módulos principais que se comunicam através de arquivos serializados:

1.  **Módulo de Engenharia de Dados e Treinamento (`treinamento_offline.py`)**:
    * **ETL:** Lê arquivos brutos (formato HidroWeb/ANA), trata falhas, remove inconsistências e consolida dados de múltiplas estações (Montante e Jusante).
    * **Feature Engineering:** Cria variáveis temporais como lags (atrasos), médias móveis (7d, 15d), acúmulos de chuva e componentes sazonais (seno/cosseno).
    * **Machine Learning:** Treina 5 modelos **XGBoost Regressor** independentes (um para cada horizonte de previsão).

2.  **Módulo de Aplicação e Visualização (`app.py`)**:
    * **Interface:** Dashboard web construído com Streamlit.
    * **Inferência:** Carrega os modelos treinados e gera previsões em tempo real baseadas nos dados mais recentes.
    * **Sistema de Alertas:** Classifica o risco (Atenção, Alerta, Inundação) baseado em percentis históricos (90%, 95%, 98%).

---

## 📂 Estrutura de Pastas

Para que o projeto funcione, a estrutura de diretórios deve ser respeitada:

```text
sistema-alerta-cheias/
│
├── 1_Dados_Brutos/          # [INPUT] Coloque aqui os CSVs baixados do HidroWeb
├── 2_Dados_Processados/     # [OUTPUT] O script salvará o dataset limpo aqui
├── 3_Modelos_Treinados/     # [OUTPUT] O script salvará os modelos .joblib e métricas aqui
│
├── app.py                   # O Dashboard (Frontend)
├── treinamento_offline.py   # O Script de Treinamento (Backend/ETL)
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação
