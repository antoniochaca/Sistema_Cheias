# 🌊 Sistema de Alerta de Cheias & Previsão Hidrológica com IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

Este projeto é uma solução de **Ciência de Dados** ponta a ponta (End-to-End) para monitoramento e previsão de níveis de rios. O sistema processa dados brutos de estações telemétricas, treina modelos de Machine Learning e disponibiliza um dashboard interativo para auxílio na tomada de decisões sobre riscos de inundação.

O modelo é capaz de prever o nível do rio na estação de Jusante com até **5 dias de antecedência (D+1 a D+5)**.

---

## Arquitetura do Projeto

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

## Estrutura de Pastas

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
```
🚀 Guia de Instalação e ExecuçãoSiga estes passos para rodar o projeto localmente.1. Clonar o Repositório e Preparar AmbienteBash# Clone este repositório
git clone [https://github.com/SEU-USUARIO/NOME-DO-REPO.git](https://github.com/SEU-USUARIO/NOME-DO-REPO.git)

# Entre na pasta
cd NOME-DO-REPO

# Instale as dependências necessárias
pip install -r requirements.txt
2. Alimentar os Dados (Etapa Crucial)O sistema espera arquivos .csv na pasta 1_Dados_Brutos/.O script busca arquivos contendo os códigos das estações no nome.Códigos configurados:Montante Fluviométrica: 56991500Montante Pluviométrica: 1941008Jusante (Alvo): 56992000Nota: Certifique-se de que os arquivos .csv dentro da pasta 1_Dados_Brutos contenham esses números em seus nomes para que o script os encontre automaticamente.3. Executar o Pipeline de TreinamentoAntes de abrir o app, processe os dados e gere os modelos. Execute o script offline:Bashpython treinamento_offline.py
O que vai acontecer:O script lerá a pasta 1_Dados_Brutos.Gerará o arquivo dados_completos_limpos.csv na pasta 2_Dados_Processados.Treinará 5 modelos XGBoost e salvará em 3_Modelos_Treinados.Exibirá no terminal a acurácia (R²) e o erro médio (MAE) de cada dia.4. Iniciar o DashboardCom os modelos gerados, inicie a aplicação visual:Bashstreamlit run app.py
O navegador abrirá automaticamente no endereço http://localhost:8501.📊 Performance do ModeloOs modelos são avaliados utilizando métricas de regressão em dados de teste (separação temporal estrita).HorizonteModeloR² (Acurácia Típica)*D+1 (Amanhã)XGBoost~0.99D+2XGBoost~0.98D+3XGBoost~0.96D+4XGBoost~0.94D+5XGBoost~0.92*Resultados podem variar dependendo do período e qualidade dos dados de entrada.🛠 Tecnologias UtilizadasLinguagem: Python 3.9+Frontend: StreamlitData Science: Pandas, Numpy, Scikit-LearnModelo Preditivo: XGBoost (Gradient Boosting)Visualização: Plotly Interactive GraphsPersistência: Joblib (Model Serialization)


