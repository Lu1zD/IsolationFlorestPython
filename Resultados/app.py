import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.ensemble import IsolationForest


st.title("Análise e Detecção de Anomalias em Logs")
#Leitura de arquivo
st.write("Selecione um arquivo de logs para iniciar a análise.")

arquivo = st.file_uploader("Escolha o arquivo de logs",type=["txt"])


if arquivo is not None:

    st.success(f"Arquivo carregado: {arquivo.name}")

    linhas = arquivo.read().decode("utf-8").splitlines()

    registros = []
    linhas_invalidas = []

    for linha in linhas:

        padrao = r"^(\w+)\s+-([\d/]+)\s+([\d:]+)\s+(.+)$"

        resultado = re.match(padrao, linha.strip())

        if resultado:
            identificador, data, hora, evento = resultado.groups()

            registros.append([identificador,data,hora,evento])

        else:
            linhas_invalidas.append(linha)

    #Criação de registros basicos
    df = pd.DataFrame(registros,columns=["identificador","data","hora","evento"])

    st.subheader("Registros encontrados")

    st.dataframe(df,use_container_width=True)

    st.write(f"Quantidade de registros válidos: {len(df)}")

    st.write(f"Quantidade de linhas inválidas: {len(linhas_invalidas)}")
    
    st.subheader("Visão geral dos dados")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Registros válidos",len(df))

    col2.metric("Linhas inválidas",len(linhas_invalidas))

    col3.metric("Identificadores",df["identificador"].nunique())

    col4.metric("Tipos de evento",df["evento"].nunique())
    
    #Distribuição de eventos por hora
    
    st.subheader("Distribuição dos eventos por hora")

    df["timestamp"] = pd.to_datetime(df["data"] + " " + df["hora"],format="%Y/%m/%d %H:%M:%S")

    df["hora_do_dia"] = df["timestamp"].dt.hour
    
    df["minuto"] = df["timestamp"].dt.minute
    
    df["segundo"] = df["timestamp"].dt.second
    
    df["dia"] = df["timestamp"].dt.date

    eventos_por_hora = (df["hora_do_dia"].value_counts().sort_index())

    st.bar_chart(eventos_por_hora)
    
    #Eventos e identificadores mais frequentes
    
    st.subheader("Eventos e identificadores mais frequentes")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 10 eventos")

        eventos_frequentes = (df["evento"].value_counts().head(10))

        st.bar_chart(eventos_frequentes)

    with col2:
        st.write("Top 10 identificadores")

        identificadores_frequentes = (df["identificador"].value_counts().head(10))

        st.bar_chart(identificadores_frequentes)
        
    #concenração temporal dos eventos 
       
    st.subheader("Concentração temporal dos 10 eventos mais frequentes")

    eventos_por_timestamp = (df["timestamp"].value_counts().head(10).sort_values())

    eventos_por_timestamp.index = (eventos_por_timestamp.index.strftime("%d/%m/%Y %H:%M:%S"))


    eventos_por_timestamp = df["timestamp"].value_counts()

    timestamp_pico = eventos_por_timestamp.index[0]
    quantidade_pico = eventos_por_timestamp.iloc[0]

    top_timestamps = (eventos_por_timestamp.head(10).sort_values())

    top_timestamps.index = (top_timestamps.index.strftime("%d/%m/%Y %H:%M:%S"))

    st.bar_chart(top_timestamps)

    st.info(f"O timestamp com maior concentração é "f"{timestamp_pico.strftime('%d/%m/%Y %H:%M:%S')}, "f"com {quantidade_pico} eventos.")
    
    st.header("Detecção de possíveis anomalias")

    st.write(
        "A etapa de Machine Learning utiliza o modelo Isolation Forest, não supervisionado para identificar registros com comportamento mais atípico no conjunto de dados.")
    st.write(    
        " Esse modelo consiste em issolar dados que o modelo julgue que sejam anomalias a partir de uma árvore. O que é ideal para esse tipo de caso em que não possuimos informações do que é anomalia ou não"
        )
    
    # FEATURES PARA O MODELO

    df["hora_sin"] = np.sin(2 * np.pi * df["hora_do_dia"] / 24)

    df["hora_cos"] = np.cos(2 * np.pi * df["hora_do_dia"] / 24)

    df["minuto_sin"] = np.sin(2 * np.pi * df["minuto"] / 60)

    df["minuto_cos"] = np.cos(2 * np.pi * df["minuto"] / 60)

    df["eventos_no_timestamp"] = (df.groupby("timestamp")["timestamp"].transform("count"))

    df["frequencia_evento"] = (df["evento"].map(df["evento"].value_counts()))

    df["frequencia_identificador"] = (df["identificador"].map(df["identificador"].value_counts()))

    # TRANSFORMAÇÕES LOGARÍTMICAS

    df["log_eventos_timestamp"] = np.log1p(df["eventos_no_timestamp"])

    df["log_frequencia_evento"] = np.log1p(df["frequencia_evento"])

    df["log_frequencia_identificador"] = np.log1p(df["frequencia_identificador"])

    # FEATURES V2

    features_v2 = ["hora_sin","hora_cos","minuto_sin","minuto_cos","log_eventos_timestamp","log_frequencia_evento","log_frequencia_identificador"]

    X_v2 = df[features_v2]

    # MODELO

    modelo_v2 = IsolationForest(n_estimators=200,contamination="auto",random_state=42)

    modelo_v2.fit(X_v2)

    df["score_v2"] = modelo_v2.decision_function(X_v2)

    # CLASSIFICAÇÃO

    limite_v2 = df["score_v2"].quantile(0.01)

    df["possivel_anomalia"] = (df["score_v2"] <= limite_v2)

    anomalias = df[df["possivel_anomalia"]].copy()

    # ==============================
    # RESULTADOS
    # ==============================

    col1, col2, col3 = st.columns(3)

    col1.metric("Registros analisados",len(df))

    col2.metric("Possíveis anomalias",len(anomalias))

    col3.metric("Percentual",f"{len(anomalias) / len(df) * 100:.2f}%")

    st.write(f"Limite de score utilizado: "f"`{limite_v2:.6f}`")
    
    st.subheader("Registros classificados como possíveis anomalias")

    anomalias_exibicao = (anomalias[["identificador","timestamp","evento","score_v2"]].sort_values("score_v2"))

    st.dataframe(anomalias_exibicao,use_container_width=True)
    
    st.subheader("Eventos mais frequentes entre as possíveis anomalias")

    eventos_anomalias = (anomalias["evento"].value_counts().head(10))

    st.bar_chart(eventos_anomalias)
    
    st.write("Os eventos apresentados são os que aparecem com maior frequência ""entre os registros classificados como possíveis anomalias pelo modelo.")
    
    st.subheader("Distribuição das possíveis anomalias por horário")

    anomalias_por_hora = (anomalias["hora_do_dia"].value_counts().sort_index())

    st.bar_chart(anomalias_por_hora)
    
    st.write("A distribuição permite identificar os horários em que os registros ""classificados como possíveis anomalias aparecem com maior frequência.")
    
    st.subheader("Timestamps com maior concentração de possíveis anomalias")

    anomalias_por_timestamp = (anomalias["timestamp"].value_counts().head(10).sort_values())

    anomalias_por_timestamp.index = (anomalias_por_timestamp.index.strftime("%d/%m/%Y %H:%M:%S"))
    
    st.bar_chart(anomalias_por_timestamp)
    
    timestamp_anomalia_pico = (anomalias["timestamp"].value_counts().idxmax())

    quantidade_anomalia_pico = (anomalias["timestamp"].value_counts().max())

    st.info(f"O timestamp com maior concentração de possíveis anomalias é "f"{timestamp_anomalia_pico.strftime('%d/%m/%Y %H:%M:%S')}, "f"com {quantidade_anomalia_pico} registros.")
    
    st.subheader("Comparação entre registros normais e possíveis anomalias")

    df_normal = df[df["score_v2"] > limite_v2].copy()

    comparacao = pd.DataFrame({"Normal": [len(df_normal),df_normal["eventos_no_timestamp"].mean(),df_normal["frequencia_evento"].mean(),df_normal["frequencia_identificador"].mean()],"Possível anomalia": [len(anomalias),anomalias["eventos_no_timestamp"].mean(),anomalias["frequencia_evento"].mean(),anomalias["frequencia_identificador"].mean()]}, index=["Quantidade de registros","Eventos no mesmo timestamp (média)","Frequência do evento (média)","Frequência do identificador (média)"])

    st.dataframe(comparacao)
    
    st.bar_chart(comparacao)
    