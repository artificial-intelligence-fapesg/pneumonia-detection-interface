"""
app.py
======
Interface de apoio ao diagnóstico de pneumonia em radiografias de tórax,
destinada a profissionais radiologistas.

Recursos:
- Cadastro clínico completo do paciente/exame.
- Upload de radiografia com animação de processamento.
- Seleção de um modelo específico OU votação (ensemble) entre modelos.
- Painel de métricas (acurácia, precisão, recall, F1-score) por modelo.
- Histórico de detecções persistido em banco de dados (SQLite).

Para rodar:
    streamlit run app.py

IMPORTANTE: esta ferramenta é um sistema de APOIO à decisão clínica.
O laudo final é sempre de responsabilidade do médico radiologista.
"""

import io
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st
from PIL import Image

import database as db
from pneumonia_detector import (
    ENSEMBLE_LABEL,
    PROCESSING_STEPS,
    DiagnosisResult,
    EnsembleResult,
    build_model_registry,
)

# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------
db.init_db()


@st.cache_resource(show_spinner=False)
def get_registry():
    return build_model_registry()


registry = get_registry()

st.set_page_config(
    page_title="Sistema de Apoio ao Diagnóstico | Pneumonia em Radiografias",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #F7F9FB; }
        .stApp { font-family: 'Segoe UI', sans-serif; }

        .clinical-header {
            background: linear-gradient(90deg, #0B3D66 0%, #145C9E 100%);
            padding: 1.4rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .clinical-header h1 { margin: 0; font-size: 1.6rem; }
        .clinical-header p { margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            border: 1px solid #E3E8EE;
            margin-bottom: 0.5rem;
        }
        .metric-card .label { font-size: 0.75rem; color: #5A6B7B; font-weight: 600; text-transform: uppercase; }
        .metric-card .value { font-size: 1.35rem; color: #0B3D66; font-weight: 700; }

        .diagnosis-box { border-radius: 12px; padding: 1.6rem; margin-top: 1rem; }
        .diagnosis-pneumonia { background-color: #FDECEC; border: 2px solid #D64550; }
        .diagnosis-normal { background-color: #E9F7EF; border: 2px solid #2E9E5B; }

        .disclaimer {
            background-color: #FFF7E6;
            border-left: 4px solid #E8A33D;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #6B4E16;
            margin-top: 1.2rem;
        }
        .vote-row {
            display: flex; justify-content: space-between; align-items: center;
            background: white; border: 1px solid #E3E8EE; border-radius: 8px;
            padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.9rem;
        }
        .vote-pneumonia { color: #D64550; font-weight: 700; }
        .vote-normal { color: #2E9E5B; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="clinical-header">
        <h1>🔍 Sistema de Apoio ao Diagnóstico Radiológico</h1>
        <p>Triagem assistida por Inteligência Artificial para detecção de pneumonia em radiografias de tórax (RX PA/AP)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral — Seleção de modelo(s) e métricas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Modelo de Análise")

    available_models = registry.list_models()
    modo_opcoes = available_models + [ENSEMBLE_LABEL]

    modo_selecionado = st.selectbox(
        "Selecione o modelo ou o modo de votação",
        options=modo_opcoes,
        index=len(modo_opcoes) - 1,  # padrão: votação
        help="Escolha um modelo específico ou combine vários modelos por votação (ensemble).",
    )

    is_ensemble = modo_selecionado == ENSEMBLE_LABEL

    if is_ensemble:
        st.caption("Modelos participantes da votação:")
        modelos_ensemble = st.multiselect(
            "Modelos incluídos na votação",
            options=available_models,
            default=available_models,
            label_visibility="collapsed",
        )
        metodo_votacao = st.radio(
            "Método de consolidação do voto",
            options=["soft", "hard"],
            format_func=lambda x: "Média de probabilidades (soft-vote)" if x == "soft" else "Maioria simples (hard-vote)",
        )
    else:
        modelos_ensemble = [modo_selecionado]
        metodo_votacao = "soft"

    st.divider()
    st.markdown("### 📊 Métricas de Desempenho")

    metrics_to_show = (
        {name: registry.get_metrics(name) for name in modelos_ensemble}
        if is_ensemble
        else {modo_selecionado: registry.get_metrics(modo_selecionado)}
    )

    for name, m in metrics_to_show.items():
        with st.expander(f"**{name}**", expanded=not is_ensemble):
            st.caption(m.get("descricao", ""))
            st.markdown(
                f"""
                <div class="metric-card"><div class="label">Acurácia</div><div class="value">{m['acuracia']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Precisão (VPP)</div><div class="value">{m['precisao']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Recall (Sensibilidade)</div><div class="value">{m['recall']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">F1-Score</div><div class="value">{m['f1_score']*100:.1f}%</div></div>
                <div class="metric-card"><div class="label">Especificidade</div><div class="value">{m['especificidade']*100:.1f}%</div></div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    total_stats = db.get_stats_summary()
    st.markdown("### 🗂️ Base de Dados")
    st.caption(
        f"{total_stats['total']} exame(s) registrados · "
        f"{total_stats['pneumonia']} com Pneumonia · {total_stats['normal']} Normal"
    )

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <strong>Uso como apoio à decisão.</strong><br>
        Este sistema não substitui a avaliação clínica e o laudo do
        médico radiologista.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Abas principais
# ---------------------------------------------------------------------------
tab_novo, tab_historico = st.tabs(["🩺 Novo Diagnóstico", "🗂️ Histórico de Detecções"])

# =============================================================================
# ABA 1 — NOVO DIAGNÓSTICO
# =============================================================================
with tab_novo:
    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("### 1. Dados do Paciente e do Exame")

        with st.form("patient_form", clear_on_submit=False):
            st.markdown("**Identificação**")
            c1, c2 = st.columns(2)
            nome = c1.text_input("Nome completo do paciente *")
            prontuario = c2.text_input("Prontuário / ID do exame")

            c3, c4, c5 = st.columns(3)
            data_nascimento = c3.date_input(
                "Data de nascimento", value=None, min_value=date(1900, 1, 1), max_value=date.today(), format="DD/MM/YYYY"
            )
            sexo = c4.selectbox("Sexo", ["Não informado", "Masculino", "Feminino", "Outro"])
            data_exame = c5.date_input("Data do exame", value=date.today(), format="DD/MM/YYYY")

            medico_solicitante = st.text_input("Médico solicitante")
            indicacao_clinica = st.text_area(
                "Indicação clínica / motivo do exame",
                placeholder="Ex.: Investigação de quadro respiratório agudo, febre há 3 dias...",
                height=70,
            )

            st.markdown("**Quadro clínico**")
            c6, c7 = st.columns(2)
            sintomas = c6.multiselect(
                "Sintomas",
                ["Febre", "Tosse", "Dispneia", "Dor torácica", "Fadiga", "Calafrios",
                 "Produção de escarro", "Confusão mental", "Outros"],
            )
            comorbidades = c7.multiselect(
                "Comorbidades / fatores de risco",
                ["Diabetes", "Hipertensão", "DPOC", "Asma", "Imunossupressão",
                 "Cardiopatia", "Tabagismo", "Obesidade", "Idade avançada (>65 anos)", "Outros"],
            )

            c8, c9, c10 = st.columns(3)
            saturacao_o2 = c8.number_input("Saturação de O₂ (%)", min_value=0, max_value=100, value=0, step=1)
            temperatura_c = c9.number_input("Temperatura corporal (°C)", min_value=30.0, max_value=43.0, value=36.5, step=0.1)
            frequencia_resp = c10.number_input("Frequência respiratória (irpm)", min_value=0, max_value=80, value=0, step=1)

            observacoes = st.text_area("Observações clínicas adicionais", height=60)

            st.markdown("**Radiografia**")
            uploaded_file = st.file_uploader(
                "Envie a imagem de radiografia de tórax (JPG ou PNG)",
                type=["jpg", "jpeg", "png"],
                help="Recomenda-se incidência PA (póstero-anterior) ou AP (ântero-posterior) de tórax.",
            )
            radiologista = st.text_input("Radiologista responsável pela análise")

            submitted = st.form_submit_button(
                "🔎 Iniciar Análise Diagnóstica", type="primary", use_container_width=True
            )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Radiografia carregada", use_container_width=True)

    with col_result:
        st.markdown("### 2. Resultado da Análise")
        result_placeholder = st.empty()

        if not uploaded_file:
            result_placeholder.info(
                "Preencha os dados do paciente, envie a radiografia e clique em "
                "**Iniciar Análise Diagnóstica** para obter o laudo assistido por IA."
            )

        if submitted:
            if not nome:
                result_placeholder.error("Informe ao menos o **nome do paciente** para prosseguir.")
            elif not uploaded_file:
                result_placeholder.error("É necessário enviar a **radiografia** para iniciar a análise.")
            elif is_ensemble and not modelos_ensemble:
                result_placeholder.error("Selecione ao menos um modelo para a votação.")
            else:
                with result_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    n_steps = len(PROCESSING_STEPS)
                    for i, step in enumerate(PROCESSING_STEPS, start=1):
                        status_text.markdown(f"🩻 **{step}**")
                        progress_bar.progress(i / n_steps)
                        time.sleep(0.3)

                    if is_ensemble:
                        result: EnsembleResult = registry.predict_ensemble(
                            image, model_names=modelos_ensemble, method=metodo_votacao
                        )
                    else:
                        single: DiagnosisResult = registry.predict_single(modo_selecionado, image)
                        result = EnsembleResult(
                            label=single.label,
                            confidence=single.confidence,
                            probabilities=single.probabilities,
                            individual_results=[single],
                            votes={single.label: 1},
                            unanimous=True,
                        )

                    progress_bar.empty()
                    status_text.empty()

                # -- Persistência no banco de dados --------------------------------
                idade = db.calculate_age(data_nascimento) if data_nascimento else None
                patient_id = db.insert_patient(
                    {
                        "nome": nome,
                        "prontuario": prontuario,
                        "data_nascimento": str(data_nascimento) if data_nascimento else None,
                        "idade": idade,
                        "sexo": sexo,
                        "data_exame": str(data_exame),
                        "medico_solicitante": medico_solicitante,
                        "indicacao_clinica": indicacao_clinica,
                        "sintomas": sintomas,
                        "comorbidades": comorbidades,
                        "tabagismo": "Sim" if "Tabagismo" in comorbidades else "Não informado",
                        "saturacao_o2": saturacao_o2 or None,
                        "temperatura_c": temperatura_c or None,
                        "frequencia_resp": frequencia_resp or None,
                        "observacoes": observacoes,
                    }
                )

                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="PNG")
                image_path = db.save_uploaded_image(buf.getvalue(), extension="png")

                votos_individuais = [
                    {"modelo": r.model_name, "label": r.label, "confidence": r.confidence,
                     "prob_normal": r.probabilities["Normal"], "prob_pneumonia": r.probabilities["Pneumonia"]}
                    for r in result.individual_results
                ]

                db.insert_detection(
                    {
                        "patient_id": patient_id,
                        "image_path": image_path,
                        "modo_analise": "votacao" if is_ensemble else "modelo_unico",
                        "modelo_utilizado": ENSEMBLE_LABEL if is_ensemble else modo_selecionado,
                        "label": result.label,
                        "confidence": result.confidence,
                        "prob_normal": result.probabilities["Normal"],
                        "prob_pneumonia": result.probabilities["Pneumonia"],
                        "votos_individuais": votos_individuais if is_ensemble else None,
                        "concordancia_modelos": int(result.unanimous) if is_ensemble else None,
                        "radiologista": radiologista,
                        "observacoes_laudo": observacoes,
                    }
                )

                # -- Exibição do resultado ------------------------------------------
                with result_placeholder.container():
                    is_pneumonia = result.label == "Pneumonia"
                    box_class = "diagnosis-pneumonia" if is_pneumonia else "diagnosis-normal"
                    icon = "🔴" if is_pneumonia else "🟢"
                    achado = (
                        "Padrão radiográfico sugestivo de <strong>consolidação/infiltrado "
                        "compatível com processo pneumônico</strong>."
                        if is_pneumonia
                        else "Padrão radiográfico <strong>sem sinais sugestivos de consolidação "
                        "pulmonar</strong> compatível com pneumonia."
                    )
                    modelo_txt = (
                        f"Votação entre {len(modelos_ensemble)} modelos ({metodo_votacao})"
                        if is_ensemble
                        else modo_selecionado
                    )

                    st.markdown(
                        f"""
                        <div class="diagnosis-box {box_class}">
                            <h3>{icon} Impressão diagnóstica assistida por IA: {result.label}</h3>
                            <p>{achado}</p>
                            <p><strong>Confiança do resultado:</strong> {result.confidence*100:.1f}%</p>
                            <p style="font-size:0.85rem; color:#5A6B7B;">
                                Paciente: {nome} · Exame: {prontuario or "não identificado"} ·
                                Modelo(s): {modelo_txt} · Processado em {datetime.now().strftime('%d/%m/%Y %H:%M')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Distribuição de Probabilidades (consolidada)")
                    prob_cols = st.columns(2)
                    with prob_cols[0]:
                        st.metric("Normal", f"{result.probabilities['Normal']*100:.1f}%")
                        st.progress(result.probabilities["Normal"])
                    with prob_cols[1]:
                        st.metric("Pneumonia", f"{result.probabilities['Pneumonia']*100:.1f}%")
                        st.progress(result.probabilities["Pneumonia"])

                    if is_ensemble:
                        st.markdown("#### Detalhamento da Votação por Modelo")
                        if result.unanimous:
                            st.success("✅ Todos os modelos concordaram quanto ao resultado.")
                        else:
                            st.warning(
                                "⚠️ Os modelos **divergiram** quanto ao resultado — recomenda-se "
                                "atenção redobrada na correlação clínica e revisão manual da imagem."
                            )

                        for r in result.individual_results:
                            vote_class = "vote-pneumonia" if r.label == "Pneumonia" else "vote-normal"
                            st.markdown(
                                f"""
                                <div class="vote-row">
                                    <span><strong>{r.model_name}</strong></span>
                                    <span class="{vote_class}">{r.label} ({r.confidence*100:.1f}%)</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        st.caption(
                            f"Votos: Normal = {result.votes.get('Normal', 0)} · "
                            f"Pneumonia = {result.votes.get('Pneumonia', 0)} · "
                            f"Método de consolidação: {'média de probabilidades' if metodo_votacao == 'soft' else 'maioria simples'}"
                        )

                    st.markdown(
                        """
                        <div class="disclaimer">
                        📌 <strong>Conduta sugerida:</strong> correlacionar este resultado com
                        a história clínica, exame físico e, se necessário, exames
                        complementares. O laudo definitivo deve ser emitido pelo médico
                        radiologista responsável.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success("Registro salvo no histórico de detecções.")

# =============================================================================
# ABA 2 — HISTÓRICO
# =============================================================================
with tab_historico:
    st.markdown("### Histórico de Detecções")

    fc1, fc2, fc3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    busca = fc1.text_input("Buscar por nome do paciente ou prontuário")
    filtro_resultado = fc2.selectbox("Resultado", ["Todos", "Normal", "Pneumonia"])
    if fc3.button("🔄 Atualizar", use_container_width=True):
        st.rerun()

    registros = db.fetch_history(nome_filtro=busca or None, resultado_filtro=filtro_resultado)

    if not registros:
        st.info("Nenhum registro encontrado. Realize uma análise na aba **Novo Diagnóstico**.")
    else:
        df = pd.DataFrame(registros)
        df_display = df[
            ["data_deteccao", "nome", "prontuario", "idade", "sexo", "label",
             "confidence", "modelo_utilizado", "concordancia_modelos"]
        ].copy()
        df_display.columns = [
            "Data/Hora", "Paciente", "Prontuário", "Idade", "Sexo",
            "Resultado", "Confiança", "Modelo(s)", "Concordância",
        ]
        df_display["Confiança"] = (df_display["Confiança"] * 100).round(1).astype(str) + "%"
        df_display["Concordância"] = df_display["Concordância"].map(
            {1: "✅ Unânime", 0: "⚠️ Divergência"}
        ).fillna("—")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("#### Detalhes do Exame")
        opcoes_detalhe = {
            f"#{r['detection_id']} · {r['nome']} · {r['data_deteccao']}": r for r in registros
        }
        escolha = st.selectbox("Selecione um exame para ver o detalhe completo", list(opcoes_detalhe.keys()))
        registro = opcoes_detalhe[escolha]

        dcol1, dcol2 = st.columns([1, 1.3])
        with dcol1:
            if registro.get("image_path"):
                try:
                    st.image(registro["image_path"], caption="Radiografia analisada", use_container_width=True)
                except Exception:
                    st.caption("Imagem não disponível.")

        with dcol2:
            st.markdown(f"**Paciente:** {registro['nome']} ({registro.get('idade', '—')} anos, {registro.get('sexo', '—')})")
            st.markdown(f"**Prontuário:** {registro.get('prontuario') or '—'}")
            st.markdown(f"**Data do exame:** {registro.get('data_exame') or '—'}")
            st.markdown(f"**Médico solicitante:** {registro.get('medico_solicitante') or '—'}")
            st.markdown(f"**Indicação clínica:** {registro.get('indicacao_clinica') or '—'}")
            st.markdown(f"**Resultado:** {registro['label']} (confiança {registro['confidence']*100:.1f}%)")
            st.markdown(f"**Modelo(s) utilizado(s):** {registro['modelo_utilizado']}")
            st.markdown(f"**Radiologista:** {registro.get('radiologista') or '—'}")

            if registro.get("votos_individuais"):
                import json
                votos = json.loads(registro["votos_individuais"])
                st.markdown("**Detalhe da votação:**")
                for v in votos:
                    st.caption(f"- {v['modelo']}: {v['label']} ({v['confidence']*100:.1f}%)")

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Sistema de apoio ao diagnóstico por imagem · Uso restrito a profissionais de saúde · "
    "Não substitui avaliação médica presencial."
)
