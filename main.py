import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d

st.set_page_config(page_title="Análise da Norma Giroscópica", layout="wide")

st.title("Análise da norma euclidiana do giroscópio")

arquivo = st.file_uploader("Carregue o arquivo TXT ou CSV", type=["txt", "csv"])

st.sidebar.header("Parâmetros da análise")

fs = st.sidebar.number_input("Frequência de interpolação (Hz)", 10, 500, 100, 10)
fc = st.sidebar.number_input("Frequência de corte do filtro passa-baixa (Hz)", 0.1, 20.0, 1.5, 0.1)
ordem = st.sidebar.number_input("Ordem do filtro Butterworth", 1, 8, 4, 1)

if arquivo is not None:

    try:
        df = pd.read_csv(arquivo, sep=r"[;,\s]+", engine="python")

        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head(), use_container_width=True)

        colunas = df.columns.tolist()

        st.sidebar.subheader("Seleção das colunas")

        col_tempo = st.sidebar.selectbox("Coluna do tempo", colunas, index=0)
        col_x = st.sidebar.selectbox("Coluna eixo X", colunas, index=1)
        col_y = st.sidebar.selectbox("Coluna eixo Y", colunas, index=2)
        col_z = st.sidebar.selectbox("Coluna eixo Z", colunas, index=3)

        unidade_tempo = st.sidebar.radio(
            "Unidade da coluna de tempo",
            ["milissegundos", "segundos"],
            index=0
        )

        t = df[col_tempo].astype(float).values
        x = df[col_x].astype(float).values
        y = df[col_y].astype(float).values
        z = df[col_z].astype(float).values

        if unidade_tempo == "milissegundos":
            t = t / 1000

        t_min = float(np.min(t))
        t_max = float(np.max(t))

        st.sidebar.subheader("Intervalo da atividade")

        inicio = st.sidebar.number_input(
            "Início da atividade (s)",
            min_value=t_min,
            max_value=t_max,
            value=4.87,
            step=0.01
        )

        final = st.sidebar.number_input(
            "Final da atividade (s)",
            min_value=t_min,
            max_value=t_max,
            value=14.59,
            step=0.01
        )

        if final <= inicio:
            st.error("O tempo final precisa ser maior que o tempo inicial.")

        else:
            mask = (t >= inicio) & (t <= final)

            t_rec = t[mask]
            x_rec = x[mask]
            y_rec = y[mask]
            z_rec = z[mask]

            if len(t_rec) < 10:
                st.error("O intervalo selecionado possui poucos pontos.")

            else:
                # =========================
                # 1. Detrend em cada eixo
                # =========================
                x_dt = detrend(x_rec)
                y_dt = detrend(y_rec)
                z_dt = detrend(z_rec)

                # =========================
                # 2. Interpolação para 100 Hz
                # =========================
                t_uniforme = np.arange(t_rec[0], t_rec[-1], 1 / fs)

                fx = interp1d(t_rec, x_dt, kind="linear", fill_value="extrapolate")
                fy = interp1d(t_rec, y_dt, kind="linear", fill_value="extrapolate")
                fz = interp1d(t_rec, z_dt, kind="linear", fill_value="extrapolate")

                x_interp = fx(t_uniforme)
                y_interp = fy(t_uniforme)
                z_interp = fz(t_uniforme)

                # =========================
                # 3. Filtro em cada eixo
                # =========================
                b, a = butter(ordem, fc / (fs / 2), btype="low")

                x_filt = filtfilt(b, a, x_interp)
                y_filt = filtfilt(b, a, y_interp)
                z_filt = filtfilt(b, a, z_interp)

                # =========================
                # 4. Norma ao final
                # =========================
                norma = np.sqrt(x_filt**2 + y_filt**2 + z_filt**2)

                # =========================
                # 5. Tempo percentual
                # =========================
                tempo_percentual = (
                    (t_uniforme - t_uniforme[0]) /
                    (t_uniforme[-1] - t_uniforme[0])
                ) * 100

                df_resultado = pd.DataFrame({
                    "tempo_s": t_uniforme,
                    "tempo_percentual": tempo_percentual,
                    "x_filtrado": x_filt,
                    "y_filtrado": y_filt,
                    "z_filtrado": z_filt,
                    "norma": norma
                })

                st.subheader("Resumo da atividade")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Início", f"{inicio:.2f} s")
                col2.metric("Final", f"{final:.2f} s")
                col3.metric("Duração", f"{final - inicio:.2f} s")
                col4.metric("Amostras após interpolação", len(df_resultado))

                st.subheader("Eixos filtrados")

                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(tempo_percentual, x_filt, label="X")
                ax1.plot(tempo_percentual, y_filt, label="Y")
                ax1.plot(tempo_percentual, z_filt, label="Z")
                ax1.set_xlabel("Tempo da atividade (%)")
                ax1.set_ylabel("Velocidade angular filtrada")
                ax1.set_title("Eixos com detrend + interpolação + filtro")
                ax1.legend()
                ax1.grid(True)

                st.pyplot(fig1)

                st.subheader("Norma euclidiana final")

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(tempo_percentual, norma)
                ax2.set_xlabel("Tempo da atividade (%)")
                ax2.set_ylabel("Norma euclidiana")
                ax2.set_title("Norma calculada após processamento dos eixos")
                ax2.grid(True)

                st.pyplot(fig2)

                st.subheader("Tabela final")
                st.dataframe(df_resultado, use_container_width=True)

                csv = df_resultado.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Baixar resultado em CSV",
                    data=csv,
                    file_name="giroscopio_processado_norma.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error("Erro ao processar o arquivo.")
        st.exception(e)

else:
    st.info("Carregue um arquivo para iniciar a análise.")
