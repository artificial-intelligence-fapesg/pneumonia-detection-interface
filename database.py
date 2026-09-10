"""
database.py
============
Camada de persistência do sistema. Usa SQLite (arquivo local, sem
necessidade de servidor) para armazenar:

- `patients`   → dados clínicos e demográficos do paciente/exame
- `detections` → cada análise realizada (modelo utilizado, resultado,
                  probabilidades, e — em caso de votação — o detalhe do
                  voto de cada modelo individual)

Não é necessário nenhum servidor de banco de dados: o arquivo
`pneumonia_app.db` é criado automaticamente na primeira execução, na
raiz do projeto. Para trocar para PostgreSQL/MySQL em produção, basta
adaptar as funções `get_connection()` e as queries desta camada — o
restante da aplicação (app.py) não precisa ser alterado, pois sempre
conversa com as funções públicas deste módulo.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "pneumonia_app.db"
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Conexão e criação das tabelas
# ---------------------------------------------------------------------------
@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Cria as tabelas do banco caso ainda não existam. Chame no início do app."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                nome                TEXT NOT NULL,
                prontuario          TEXT,
                data_nascimento     TEXT,
                idade               INTEGER,
                sexo                TEXT,
                data_exame          TEXT,
                medico_solicitante  TEXT,
                indicacao_clinica   TEXT,
                sintomas            TEXT,      -- JSON list
                comorbidades        TEXT,      -- JSON list
                tabagismo           TEXT,
                saturacao_o2        REAL,
                temperatura_c       REAL,
                frequencia_resp     INTEGER,
                observacoes         TEXT,
                criado_em           TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id              INTEGER NOT NULL,
                image_path              TEXT,
                modo_analise            TEXT NOT NULL,   -- "modelo_unico" ou "votacao"
                modelo_utilizado        TEXT NOT NULL,   -- nome do modelo, ou "Votação (Ensemble)"
                label                   TEXT NOT NULL,   -- "Normal" ou "Pneumonia"
                confidence              REAL NOT NULL,
                prob_normal             REAL NOT NULL,
                prob_pneumonia          REAL NOT NULL,
                votos_individuais       TEXT,             -- JSON com detalhe de cada modelo (se votação)
                concordancia_modelos    INTEGER,           -- 1 se todos os modelos concordaram, 0 se não, NULL se modelo único
                radiologista            TEXT,
                observacoes_laudo       TEXT,
                criado_em               TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
            """
        )


# ---------------------------------------------------------------------------
# Pacientes
# ---------------------------------------------------------------------------
def insert_patient(data: dict) -> int:
    """Insere um paciente/exame e retorna o id gerado."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO patients (
                nome, prontuario, data_nascimento, idade, sexo, data_exame,
                medico_solicitante, indicacao_clinica, sintomas, comorbidades,
                tabagismo, saturacao_o2, temperatura_c, frequencia_resp,
                observacoes, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("nome"),
                data.get("prontuario"),
                data.get("data_nascimento"),
                data.get("idade"),
                data.get("sexo"),
                data.get("data_exame"),
                data.get("medico_solicitante"),
                data.get("indicacao_clinica"),
                json.dumps(data.get("sintomas", []), ensure_ascii=False),
                json.dumps(data.get("comorbidades", []), ensure_ascii=False),
                data.get("tabagismo"),
                data.get("saturacao_o2"),
                data.get("temperatura_c"),
                data.get("frequencia_resp"),
                data.get("observacoes"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Detecções
# ---------------------------------------------------------------------------
def save_uploaded_image(image_bytes: bytes, extension: str = "png") -> str:
    """Salva a imagem em disco (pasta uploads/) e retorna o caminho relativo."""
    filename = f"{uuid.uuid4().hex}.{extension}"
    filepath = UPLOADS_DIR / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return str(filepath)


def insert_detection(data: dict) -> int:
    """Insere um registro de detecção e retorna o id gerado."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO detections (
                patient_id, image_path, modo_analise, modelo_utilizado,
                label, confidence, prob_normal, prob_pneumonia,
                votos_individuais, concordancia_modelos, radiologista,
                observacoes_laudo, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["patient_id"],
                data.get("image_path"),
                data["modo_analise"],
                data["modelo_utilizado"],
                data["label"],
                data["confidence"],
                data["prob_normal"],
                data["prob_pneumonia"],
                json.dumps(data.get("votos_individuais"), ensure_ascii=False)
                if data.get("votos_individuais") is not None
                else None,
                data.get("concordancia_modelos"),
                data.get("radiologista"),
                data.get("observacoes_laudo"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cur.lastrowid


def fetch_history(
    nome_filtro: Optional[str] = None,
    resultado_filtro: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Retorna o histórico de detecções (join com dados do paciente), mais recentes primeiro."""
    query = """
        SELECT
            d.id AS detection_id, d.image_path, d.modo_analise, d.modelo_utilizado,
            d.label, d.confidence, d.prob_normal, d.prob_pneumonia,
            d.votos_individuais, d.concordancia_modelos, d.radiologista,
            d.observacoes_laudo, d.criado_em AS data_deteccao,
            p.nome, p.prontuario, p.idade, p.sexo, p.data_exame,
            p.medico_solicitante, p.indicacao_clinica, p.sintomas,
            p.comorbidades
        FROM detections d
        JOIN patients p ON p.id = d.patient_id
        WHERE 1=1
    """
    params: list = []
    if nome_filtro:
        query += " AND (p.nome LIKE ? OR p.prontuario LIKE ?)"
        params.extend([f"%{nome_filtro}%", f"%{nome_filtro}%"])
    if resultado_filtro and resultado_filtro != "Todos":
        query += " AND d.label = ?"
        params.append(resultado_filtro)
    query += " ORDER BY d.criado_em DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_stats_summary() -> dict:
    """Estatísticas rápidas para o painel (total de exames, distribuição de resultados)."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"]
        pneumonia = conn.execute(
            "SELECT COUNT(*) AS c FROM detections WHERE label = 'Pneumonia'"
        ).fetchone()["c"]
        normal = conn.execute(
            "SELECT COUNT(*) AS c FROM detections WHERE label = 'Normal'"
        ).fetchone()["c"]
        return {"total": total, "pneumonia": pneumonia, "normal": normal}


def calculate_age(birth_date: date, reference_date: Optional[date] = None) -> Optional[int]:
    if not birth_date:
        return None
    reference_date = reference_date or date.today()
    years = reference_date.year - birth_date.year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years
