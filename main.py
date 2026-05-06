import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d

st.set_page_config(page_title="Análise Giroscópio", layout="wide")

st.title("Processamento de dados do giroscópio com padronização temporal")

arquivo = st.file_uploader("Carregue o arquivo TXT ou CSV", type=["txt", "csv"])

st.sidebar.header("Parâmetros")

fs = st.sidebar.number_input(
    "Frequência de interpolação inicial (Hz)",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

fc = st.sidebar.number_input(
    "Frequência de corte do filtro passa-baixa (Hz)",
    min_value=0.1,
    max_value=20.0,
    value=1.5,
    step=0.1
)

ordem = st.sidebar.number_input(
    "Ordem do filtro Butterworth",
    min_value=1,
    max_value=8,
    value=4,
    step=1
)

n_pontos = st.sidebar.number_input(
    "Número de pontos padronizados",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)

unidade = st.sidebar.radio(
    "Unidade do tempo no arquivo",
    ["ms", "s"]
)

if arquivo is not None:

    try:
        df = pd.read_csv(
            arquivo,
            sep=r"[;,\s]+",
            engine="python"
        )

        st.subheader("Pré-visualização do arquivo")
        st.dataframe(df.head(), use_container_width=True)

        if df.shape[1] < 4:
            st.error("O arquivo precisa ter pelo menos 4 colunas: tempo, X, Y, Z.")
            st.stop()

        t = df.iloc[:, 0].astype(float).values
        x = df.iloc[:, 1].astype(float).values
        y = df.iloc[:, 2].astype(float).values
        z = df.iloc[:, 3].astype(float).values

        if unidade == "ms":
            t = t / 1000.0

        st.sidebar.subheader("Recorte da atividade")

        inicio = st.sidebar.number_input(
            "Início da atividade (s)",
            min_value=float(np.min(t)),
            max_value=float(np.max(t)),
            value=float(np.min(t)),
            step=0.01
        )

        final = st.sidebar.number_input(
            "Final da atividade (s)",
            min_value=float(np.min(t)),
            max_value=float(np.max(t)),
            value=float(np.max(t)),
            step=0.01
        )

        if final <= inicio:
            st.error("O tempo final deve ser maior que o tempo inicial.")
            st.stop()

        mask = (t >= inicio) & (t <= final)

        t_rec = t[mask]
        x_rec = x[mask]
        y_rec = y[mask]
        z_rec = z[mask]

        if len(t_rec) < 10:
            st.error("Poucos pontos no intervalo selecionado.")
            st.stop()

        x_dt = detrend(x_rec)
        y_dt = detrend(y_rec)
        z_dt = detrend(z_rec)

        t_uniforme = np.arange(
            t_rec[0],
            t_rec[-1],
            1 / fs
        )

        if len(t_uniforme) < 10:
            st.error("A interpolação gerou poucos pontos. Verifique o intervalo selecionado.")
            st.stop()

        fx = interp1d(t_rec, x_dt, kind="linear", fill_value="extrapolate")
        fy = interp1d(t_rec, y_dt, kind="linear", fill_value="extrapolate")
        fz = interp1d(t_rec, z_dt, kind="linear", fill_value="extrapolate")

        x_interp = fx(t_uniforme)
        y_interp = fy(t_uniforme)
        z_interp = fz(t_uniforme)

        nyquist = fs / 2

        if fc >= nyquist:
            st.error("A frequência de corte deve ser menor que a frequência de Nyquist.")
            st.stop()

        b, a = butter(
            int(ordem),
            fc / nyquist,
            btype="low"
        )

        x_filt = filtfilt(b, a, x_interp)
        y_filt = filtfilt(b, a, y_interp)
        z_filt = filtfilt(b, a, z_interp)

        norma = np.sqrt(
            x_filt**2 +
            y_filt**2 +
            z_filt**2
        )

        tempo_percentual = (
            (t_uniforme - t_uniforme[0]) /
            (t_uniforme[-1] - t_uniforme[0])
        ) * 100

        tempo_padronizado = np.linspace(0, 100, int(n_pontos))
        quantil = np.linspace(0, 1, int(n_pontos))

        x_pad = np.interp(tempo_padronizado, tempo_percentual, x_filt)
        y_pad = np.interp(tempo_padronizado, tempo_percentual, y_filt)
        z_pad = np.interp(tempo_padronizado, tempo_percentual, z_filt)
        norma_pad = np.interp(tempo_padronizado, tempo_percentual, norma)

        df_final = pd.DataFrame({
            "quantil": quantil,
            "tempo_%": tempo_padronizado,
            "x_filtrado": x_pad,
            "y_filtrado": y_pad,
            "z_filtrado": z_pad,
            "norma": norma_pad
        })

        st.subheader("Resumo do processamento")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Início", f"{inicio:.2f} s")
        c2.metric("Final", f"{final:.2f} s")
        c3.metric("Duração", f"{final - inicio:.2f} s")
        c4.metric("Pontos originais recortados", len(t_rec))
        c5.metric("Pontos exportados", len(df_final))

        st.subheader("Eixos filtrados e padronizados")

        fig1, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(tempo_padronizado, x_pad, label="X")
        ax1.plot(tempo_padronizado, y_pad, label="Y")
        ax1.plot(tempo_padronizado, z_pad, label="Z")

        ax1.set_xlabel("Tempo normalizado da atividade (%)")
        ax1.set_ylabel("Velocidade angular filtrada")
        ax1.legend()
        ax1.grid(True)

        st.pyplot(fig1)

        st.subheader("Norma euclidiana padronizada")

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.plot(tempo_padronizado, norma_pad)

        ax2.set_xlabel("Tempo normalizado da atividade (%)")
        ax2.set_ylabel("Norma euclidiana")
        ax2.grid(True)

        st.pyplot(fig2)

        st.subheader("Dados processados e padronizados")
        st.dataframe(df_final, use_container_width=True)

        csv = df_final.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Baixar CSV padronizado",
            data=csv,
            file_name="dados_padronizados_1500_pontos.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Erro no processamento.")
        st.exception(e)

else:
    st.info("Carregue um arquivo para começar.")
