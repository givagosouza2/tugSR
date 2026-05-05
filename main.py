import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d

st.set_page_config(page_title="Análise Giroscópio", layout="wide")

st.title("Processamento de dados do giroscópio")

arquivo = st.file_uploader("Carregue o arquivo TXT ou CSV", type=["txt", "csv"])

# =========================
# Parâmetros
# =========================
st.sidebar.header("Parâmetros")

fs = st.sidebar.number_input("Frequência de interpolação (Hz)", 10, 500, 100)
fc = st.sidebar.number_input("Frequência de corte (Hz)", 0.1, 20.0, 1.5)
ordem = st.sidebar.number_input("Ordem do filtro", 1, 8, 4)

if arquivo is not None:

    try:
        # =========================
        # Leitura robusta
        # =========================
        df = pd.read_csv(arquivo, sep=r"[;,\s]+", engine="python")

        st.subheader("Pré-visualização")
        st.dataframe(df.head(), use_container_width=True)

        # =========================
        # Verificação
        # =========================
        if df.shape[1] < 4:
            st.error("O arquivo precisa ter pelo menos 4 colunas: tempo, X, Y, Z.")
            st.stop()

        # =========================
        # Extração direta (fixa)
        # =========================
        t = df.iloc[:, 0].astype(float).values
        x = df.iloc[:, 1].astype(float).values
        y = df.iloc[:, 2].astype(float).values
        z = df.iloc[:, 3].astype(float).values

        # =========================
        # Unidade do tempo
        # =========================
        unidade = st.sidebar.radio("Unidade do tempo", ["ms", "s"])

        if unidade == "ms":
            t = t / 1000

        # =========================
        # Intervalo da atividade
        # =========================
        st.sidebar.subheader("Recorte da atividade")

        inicio = st.sidebar.number_input("Início (s)", float(np.min(t)), float(np.max(t)), 4.87)
        final = st.sidebar.number_input("Final (s)", float(np.min(t)), float(np.max(t)), 14.59)

        if final <= inicio:
            st.error("Final deve ser maior que início.")
            st.stop()

        mask = (t >= inicio) & (t <= final)

        t_rec = t[mask]
        x_rec = x[mask]
        y_rec = y[mask]
        z_rec = z[mask]

        if len(t_rec) < 10:
            st.error("Poucos pontos no intervalo.")
            st.stop()

        # =========================
        # 1. Detrend em cada eixo
        # =========================
        x_dt = detrend(x_rec)
        y_dt = detrend(y_rec)
        z_dt = detrend(z_rec)

        # =========================
        # 2. Interpolação 100 Hz
        # =========================
        t_uniforme = np.arange(t_rec[0], t_rec[-1], 1 / fs)

        fx = interp1d(t_rec, x_dt, kind="linear", fill_value="extrapolate")
        fy = interp1d(t_rec, y_dt, kind="linear", fill_value="extrapolate")
        fz = interp1d(t_rec, z_dt, kind="linear", fill_value="extrapolate")

        x_interp = fx(t_uniforme)
        y_interp = fy(t_uniforme)
        z_interp = fz(t_uniforme)

        # =========================
        # 3. Filtro 1.5 Hz
        # =========================
        b, a = butter(ordem, fc / (fs / 2), btype="low")

        x_filt = filtfilt(b, a, x_interp)
        y_filt = filtfilt(b, a, y_interp)
        z_filt = filtfilt(b, a, z_interp)

        # =========================
        # 4. Norma euclidiana
        # =========================
        norma = np.sqrt(x_filt**2 + y_filt**2 + z_filt**2)

        # =========================
        # 5. Tempo normalizado
        # =========================
        tempo_percentual = (
            (t_uniforme - t_uniforme[0]) /
            (t_uniforme[-1] - t_uniforme[0])
        ) * 100

        # =========================
        # DataFrame final
        # =========================
        df_final = pd.DataFrame({
            "tempo_s": t_uniforme,
            "tempo_%": tempo_percentual,
            "x_filtrado": x_filt,
            "y_filtrado": y_filt,
            "z_filtrado": z_filt,
            "norma": norma
        })

        # =========================
        # Resumo
        # =========================
        st.subheader("Resumo")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Início", f"{inicio:.2f} s")
        c2.metric("Final", f"{final:.2f} s")
        c3.metric("Duração", f"{final - inicio:.2f} s")
        c4.metric("Amostras", len(df_final))

        # =========================
        # Plot eixos
        # =========================
        st.subheader("Eixos filtrados")

        fig1, ax1 = plt.subplots(figsize=(10,4))
        ax1.plot(tempo_percentual, x_filt, label="X")
        ax1.plot(tempo_percentual, y_filt, label="Y")
        ax1.plot(tempo_percentual, z_filt, label="Z")
        ax1.set_xlabel("Tempo (%)")
        ax1.set_ylabel("Velocidade angular")
        ax1.legend()
        ax1.grid()

        st.pyplot(fig1)

        # =========================
        # Plot norma
        # =========================
        st.subheader("Norma euclidiana")

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(tempo_percentual, norma)
        ax2.set_xlabel("Tempo (%)")
        ax2.set_ylabel("Norma")
        ax2.grid()

        st.pyplot(fig2)

        # =========================
        # Tabela
        # =========================
        st.subheader("Dados processados")
        st.dataframe(df_final, use_container_width=True)

        # =========================
        # Download
        # =========================
        csv = df_final.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Baixar CSV",
            csv,
            "dados_processados.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Erro no processamento")
        st.exception(e)

else:
    st.info("Carregue um arquivo para começar.")
