# app.py
# -*- coding: utf-8 -*-
# Streamlit Lab Response Time UI (Mockup 1:1)
# Autor: ChatGPT

import math
import random
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# 0) Configuración general
# =========================
st.set_page_config(page_title="Dashboard de Laboratorio - TaT", layout="wide")

# Paleta por secciones y elementos
CLR_RESUMEN = "#0ea5e9"     # celeste
CLR_KPI     = "#10b981"     # verde
CLR_HOJAS   = "#f59e0b"     # ámbar
CLR_PANEL   = "#111827"     # gris muy oscuro para títulos
PALETTE     = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
OK_COLOR    = "#10B981"
GRAY        = "#E5E7EB"

# ========= PARÁMETROS / SLA =========
STAGE_DEADLINES = {
    "Ingreso": 5,                    # min
    "Pesaje": 70,
    "Ataque": 300,
    "Lectura": 60,
    "Reporte": 60,
    "Validación de resultados": 5,
}
SHEET_DEADLINE_LABEL = "8h 20m"
SLA_HOURS = 8 + 20/60        # 8.33 h
PRE_ALERT_HOURS = 100
FINAL_ALERT_HOURS = 120

ANALISIS_BY_TYPE = {
    "Metálico": ["Impurezas", "Azufre", "Oxígeno"],
    "No Metálico": ["Cobre", "Impurezas_AAS", "Impurezas_ICP", "Azufre", "Ensayo a fuego"],
}

MIN = 60_000

def fmt_date_key(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def fmt_sheet_name(d: datetime, n: int) -> str:
    return f"{d.strftime('%d-%m-%Y')}/{n}"

# =========================
# 1) Modelos de datos
# =========================
@dataclass
class Stage:
    name: str
    start: Optional[int] = None  # epoch ms
    end: Optional[int] = None    # epoch ms
    completed: bool = False

@dataclass
class Sample:
    id: str
    name: str
    addedAt: datetime
    type: str
    analyst: str
    analisis: str
    stages: List[Stage]

@dataclass
class Sheet:
    id: str
    name: str
    createdAt: datetime
    dateKey: str
    type: str
    samples: List[Sample] = field(default_factory=list)

DEFAULT_STAGES = [s for s in STAGE_DEADLINES.keys()]

def r(a: float, b: float) -> float:
    return random.random() * (b - a) + a

def pick_weighted(over_fn, under_fn, weight_over: float = 0.7) -> float:
    return over_fn() if random.random() < weight_over else under_fn()

def generate_simulated_stages(base_dt: datetime, profile: int = 0, sample_offset_min: int = 0) -> List[Stage]:
    """Genera etapas con gaps y duraciones (promedios mayores en Pesaje, Ataque, Lectura)."""
    t = base_dt + timedelta(minutes=sample_offset_min)
    t_ms = int(t.timestamp() * 1000)
    stages: List[Stage] = []

    # Ingreso (4-6 min)
    start = t_ms
    end = t_ms + int(r(4, 6) * MIN)
    stages.append(Stage("Ingreso", start, end, True))
    t_ms = end + 60 * MIN  # gap 1h

    # Pesaje (promedio > 70)
    start = t_ms
    pesaje_dur = pick_weighted(lambda: r(90, 140), lambda: r(45, 70))
    end = t_ms + int(pesaje_dur * MIN)
    stages.append(Stage("Pesaje", start, end, True))
    t_ms = end + 60 * MIN  # gap 1h

    # Ataque (promedio > 300)
    start = t_ms
    ataque_dur = pick_weighted(lambda: r(320, 480), lambda: r(260, 300))
    end = t_ms + int(ataque_dur * MIN)
    stages.append(Stage("Ataque", start, end, True))

    # Gap 24-48 h
    t_ms = end + int(r(24, 48) * 60 * MIN)

    # Lectura (promedio > 60)
    start = t_ms
    lectura_dur = pick_weighted(lambda: r(70, 180), lambda: r(40, 60))
    end = t_ms + int(lectura_dur * MIN)
    stages.append(Stage("Lectura", start, end, True))
    t_ms = end + 120 * MIN  # gap 2h

    # Reporte (~<60)
    start = t_ms
    end = t_ms + int(r(40, 65) * MIN)
    stages.append(Stage("Reporte", start, end, True))

    # Validación (~5)
    start = end
    end = start + int(r(4, 6) * MIN)
    stages.append(Stage("Validación de resultados", start, end, True))

    # Perfiles de avance
    if profile == 1:
        stages[4] = Stage("Reporte")
        stages[5] = Stage("Validación de resultados")
    elif profile == 2:
        stages[3] = Stage("Lectura")
        stages[4] = Stage("Reporte")
        stages[5] = Stage("Validación de resultados")
    elif profile == 3:
        stages[2] = Stage("Ataque")
        stages[3] = Stage("Lectura")
        stages[4] = Stage("Reporte")
        stages[5] = Stage("Validación de resultados")

    return stages

# =========================
# 2) Estado inicial: 10 hojas con fechas correlativas y tipo único por hoja
#    6 No Metálicas (primeras) y 4 Metálicas (últimas)
# =========================
def init_state():
    if "sheets" not in st.session_state:
        today = datetime.now().date()
        base_day = today - timedelta(days=9)  # 10 días correlativos
        sheets: List[Sheet] = []
        seq_per_day: Dict[str, int] = {}
        for i in range(10):
            sheet_date = base_day + timedelta(days=i)
            # hora aleatoria del día
            created_at = datetime.combine(sheet_date, dtime(hour=random.randint(7, 18), minute=random.randint(0, 59)))
            date_key = fmt_date_key(created_at)

            # Definir tipo de la hoja
            sheet_type = "No Metálico" if i < 6 else "Metálico"

            # Numeración por día (permite repetición de fechas con distinto contador)
            seq_per_day[date_key] = seq_per_day.get(date_key, 0) + 1
            name = fmt_sheet_name(created_at, seq_per_day[date_key])

            # Crear muestras todas del mismo tipo para esta hoja
            samples = []
            samples_count = random.randint(5, 8)
            for j in range(samples_count):
                profile = i % 4
                stages = generate_simulated_stages(created_at, profile=profile, sample_offset_min=j*10)
                an_list = ANALISIS_BY_TYPE.get(sheet_type, ["-"])
                samples.append(Sample(
                    id=str(j+1),
                    name=f"Muestra {j+1}",
                    addedAt=created_at,
                    stages=stages,
                    type=sheet_type,
                    analyst=f"Analista {((i+j) % 6) + 1}",
                    analisis=an_list[j % len(an_list)],
                ))

            s = Sheet(
                id=f"sheet_{i}_{random.randint(1000,9999)}",
                name=name,
                createdAt=created_at,
                dateKey=date_key,
                type=sheet_type,
                samples=samples
            )
            sheets.append(s)

        # Ordenar por fecha descendente (más reciente primero)
        sheets.sort(key=lambda x: x.createdAt, reverse=True)

        st.session_state.sheets = sheets
        st.session_state.selected_sheet_id = sheets[0].id if sheets else ""
        st.session_state.alert_view = "final"  # 'final' | 'pre' | 'sla'

    if "bulk_rows" not in st.session_state:
        st.session_state.bulk_rows = [{
            "id": "",
            "name": "",
            "type": "Metálico",
            "analyst": "",
            "analisis": ANALISIS_BY_TYPE["Metálico"][0]
        }]

init_state()

def get_sheets() -> List[Sheet]:
    return st.session_state.sheets

def set_sheets(new_list: List[Sheet]):
    st.session_state.sheets = new_list

# =========================
# 3) Cálculos / Helpers
# =========================
def minutes_to_hhmm(mins: float) -> str:
    if not np.isfinite(mins):
        return "N/A"
    h = int(mins // 60)
    m = int(round(mins % 60))
    return f"{h}h {m}m"

def sample_progress(stages: List[Stage]) -> float:
    total_weight = 0.0
    done_weight  = 0.0
    for stg in stages:
        w = STAGE_DEADLINES.get(stg.name, 1)
        total_weight += w
        if stg.completed and stg.start and stg.end:
            done_weight += w
    return (done_weight / total_weight) * 100.0 if total_weight > 0 else 0.0

def sheet_progress(sheet: Sheet) -> float:
    if not sheet.samples:
        return 0.0
    return np.mean([sample_progress(s.stages) for s in sheet.samples]).item()

def get_elapsed_str(dt: datetime) -> str:
    diff = datetime.now() - dt
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"

def average_turnaround_minutes(sheets: List[Sheet]) -> float:
    mins = []
    for sh in sheets:
        for s in sh.samples:
            ing = next((x for x in s.stages if x.name == "Ingreso"), None)
            val = next((x for x in s.stages if x.name == "Validación de resultados"), None)
            if ing and val and ing.start and val.end and val.end > ing.start:
                mins.append((val.end - ing.start) / 60000.0)
    return float(np.mean(mins)) if mins else float("nan")

def count_completed_validation(sheets: List[Sheet]) -> int:
    c = 0
    for sh in sheets:
        for s in sh.samples:
            ing = next((x for x in s.stages if x.name == "Ingreso"), None)
            val = next((x for x in s.stages if x.name == "Validación de resultados"), None)
            if ing and val and ing.start and val.end and val.end > ing.start:
                c += 1
    return c

def calculate_stage_durations(sheets: List[Sheet]) -> pd.DataFrame:
    durations: Dict[str, List[float]] = {}
    for sh in sheets:
        for s in sh.samples:
            for stg in s.stages:
                if stg.start and stg.end:
                    durations.setdefault(stg.name, []).append((stg.end - stg.start) / 60000.0)
    rows = []
    for name in STAGE_DEADLINES.keys():
        arr = durations.get(name, [])
        prom = round(float(np.mean(arr)), 1) if arr else 0.0
        rows.append({"etapa": name, "promedio": prom, "sla": STAGE_DEADLINES[name]})
    return pd.DataFrame(rows)

def calculate_interstage_gaps(sheets: List[Sheet]) -> pd.DataFrame:
    pairs = []
    order = list(STAGE_DEADLINES.keys())
    for i in range(len(order)-1):
        pairs.append((order[i], order[i+1]))

    gaps: Dict[str, List[float]] = {}
    for sh in sheets:
        for s in sh.samples:
            for (a, b) in pairs:
                A = next((x for x in s.stages if x.name == a), None)
                B = next((x for x in s.stages if x.name == b), None)
                if A and B and A.end and B.start:
                    gap_min = max(0.0, (B.start - A.end)/60000.0)
                    gaps.setdefault(f"{a} -> {b}", []).append(gap_min)

    rows = []
    for (a, b) in pairs:
        key = f"{a} -> {b}"
        arr = gaps.get(key, [])
        prom = round(float(np.mean(arr)), 1) if arr else 0.0
        rows.append({"transicion": key, "gapPromedio": prom, "gapHoras": round(prom/60.0, 2)})
    return pd.DataFrame(rows)

def compute_compliance_by_stage(sheets: List[Sheet]) -> pd.DataFrame:
    passed: Dict[str,int] = {}
    total: Dict[str,int] = {}
    for sh in sheets:
        for s in sh.samples:
            for stg in s.stages:
                sla = STAGE_DEADLINES.get(stg.name)
                if sla and stg.start and stg.end:
                    dur = (stg.end - stg.start) / 60000.0
                    total[stg.name] = total.get(stg.name, 0) + 1
                    if dur <= sla:
                        passed[stg.name] = passed.get(stg.name, 0) + 1
    rows = []
    for name in STAGE_DEADLINES.keys():
        t = total.get(name, 0)
        v = round((passed.get(name, 0) / t) * 100) if t > 0 else 0
        rows.append({"name": name, "value": v})
    return pd.DataFrame(rows)

def compliance_general(df_comp: pd.DataFrame) -> int:
    if df_comp.empty:
        return 0
    return int(round(df_comp["value"].mean()))

# =========================
# 4) Mutaciones (Registro rápido e ingreso)
# =========================
def register_stage_time_for_sheet(sheet_id: str, stage_name: str, action: str, analyst_name: str, sample_type: str, analisis: str):
    sheets = get_sheets()
    now_ms = int(datetime.now().timestamp() * 1000)
    new_list: List[Sheet] = []
    for sh in sheets:
        if sh.id != sheet_id:
            new_list.append(sh)
            continue
        new_samples = []
        for s in sh.samples:
            if s.type != sample_type or s.analisis != analisis:
                new_samples.append(s)
                continue
            if analyst_name:
                s.analyst = analyst_name
            new_stages = []
            for stg in s.stages:
                if stg.name == stage_name:
                    if action == "inicio":
                        if stg.start is None:
                            stg.start = now_ms
                    else:
                        stg.end = now_ms
                        stg.completed = True
                new_stages.append(stg)
            s.stages = new_stages
            new_samples.append(s)
        sh.samples = new_samples
        new_list.append(sh)
    set_sheets(new_list)

def add_bulk_row():
    st.session_state.bulk_rows.append({
        "id": "",
        "name": "",
        "type": "Metálico",
        "analyst": "",
        "analisis": ANALISIS_BY_TYPE["Metálico"][0]
    })

def remove_bulk_row(idx: int):
    st.session_state.bulk_rows.pop(idx)

def save_batch():
    rows = []
    for r in st.session_state.bulk_rows:
        rid = str(r["id"]).strip()
        if rid and r["type"] in ("Metálico", "No Metálico"):
            rows.append({
                "id": rid,
                "name": (r["name"] or "").strip(),
                "type": r["type"],
                "analyst": (r["analyst"] or "").strip(),
                "analisis": r["analisis"] or (ANALISIS_BY_TYPE.get(r["type"], [""])[0] or ""),
            })
    if not rows:
        return

    created_at = datetime.now()
    date_key = fmt_date_key(created_at)
    # contar hojas existentes del mismo día
    count = 1 + sum(1 for sh in get_sheets() if sh.dateKey == date_key)

    samples: List[Sample] = []
    for j, rr in enumerate(rows):
        profile = count % 4
        stages = generate_simulated_stages(created_at, profile=profile, sample_offset_min=j*10)
        samples.append(Sample(
            id=rr["id"],
            name=rr["name"] or f"Muestra {rr['id']}",
            addedAt=created_at,
            stages=stages,
            type=rows[0]["type"],              # toda la hoja del mismo tipo
            analyst=rr["analyst"] or "-",
            analisis=rr["analisis"],
        ))
    new_sheet = Sheet(
        id=f"sheet_{random.randint(10000,99999)}",
        name=fmt_sheet_name(created_at, count),
        createdAt=created_at,
        dateKey=date_key,
        type=rows[0]["type"],
        samples=samples
    )
    set_sheets([new_sheet] + get_sheets())
    st.session_state.selected_sheet_id = new_sheet.id
    st.session_state.bulk_rows = [{
        "id": "",
        "name": "",
        "type": "Metálico",
        "analyst": "",
        "analisis": ANALISIS_BY_TYPE["Metálico"][0]
    }]

# =========================
# 5) UI Helpers
# =========================
def section_header(title: str, color: str):
    st.markdown(f"""
    <div style="background:{color};padding:10px 14px;border-radius:12px;margin:6px 0 12px 0;">
        <h3 style="margin:0;color:white;">{title}</h3>
    </div>
    """, unsafe_allow_html=True)

def donut(value: float, label_center: str = "", show_remaining: bool = True) -> go.Figure:
    data = [value, 100 - value] if show_remaining else [value]
    colors = [OK_COLOR, GRAY] if show_remaining else [OK_COLOR]
    fig = go.Figure(data=[go.Pie(
        labels=["Cumplido", "Incumplido"] if show_remaining else ["Cumplido"],
        values=data,
        hole=.6,
        textinfo="none",
        marker=dict(colors=colors)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    fig.add_annotation(text=f"<b>{int(round(value))}%</b>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=22))
    if label_center:
        fig.add_annotation(text=f"<br><span style='font-size:12px;color:#6b7280'>{label_center}</span>",
                           x=0.5, y=0.45, showarrow=False)
    return fig

# =========================
# 6) Layout principal
# =========================
st.markdown(f"<h1 style='color:{CLR_PANEL}'>Dashboard de Laboratorio</h1>", unsafe_allow_html=True)

tabs = st.tabs(["Resumen", "KPI", "Hojas de trabajo", "Ingreso"])

sheets = get_sheets()

# ========= TAB 0: Resumen =========
with tabs[0]:
    section_header("Banner de alerta", CLR_RESUMEN)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Hojas de riesgo")
    with col_right:
        view = st.selectbox("Umbral", options=[("final", f"Final (≥ {FINAL_ALERT_HOURS} h)"),
                                               ("pre", f"Previa (≥ {PRE_ALERT_HOURS} h)"),
                                               ("sla", f"SLA (≥ {SLA_HOURS:.2f} h)")],
                            format_func=lambda x: x[1], key="alert_view_select")
        st.session_state.alert_view = view[0]

    sheets_with_age = []
    now = datetime.now()
    for sh in sheets:
        hours_since = (now - sh.createdAt).total_seconds() / 3600.0
        sheets_with_age.append((sh, hours_since))

    min_hours = FINAL_ALERT_HOURS if st.session_state.alert_view == "final" else PRE_ALERT_HOURS if st.session_state.alert_view == "pre" else SLA_HOURS
    in_alert = [(sh, hrs) for (sh, hrs) in sheets_with_age if hrs >= min_hours]
    in_alert.sort(key=lambda x: x[1], reverse=True)

    if in_alert:
        st.info(f"Se encontraron {len(in_alert)} hoja(s) que cumplen el criterio seleccionado.")
        st.selectbox(
            "Hojas en riesgo",
            options=in_alert,
            format_func=lambda x: f"{x[0].name} — {int(math.floor(x[1]))}h {int(round((x[1]%1)*60))}m",
            key="alert_dropdown"
        )
    else:
        st.success("No hay hojas que cumplan el criterio seleccionado.")

    st.markdown("")
    section_header("Registro Rápido de Avance", CLR_RESUMEN)
    opts = [(sh.id, sh.name) for sh in sheets]
    selected_id = opts[0][0] if opts else ""
    sel = st.selectbox("Hoja", options=opts, format_func=lambda x: x[1], key="selected_sheet_resumen")
    selected_id = sel[0] if isinstance(sel, tuple) else sel

    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
    with c1:
        sample_type = st.selectbox("Tipo de muestra", ["Metálico", "No Metálico"], key="rr_tipo_res")
    with c2:
        analisis = st.selectbox("Análisis", ANALISIS_BY_TYPE.get(sample_type, ["-"]), key="rr_an_res")
    with c3:
        stage_action = st.selectbox("Etapa", list(STAGE_DEADLINES.keys()), key="rr_etapa_res")
    with c4:
        action_type = st.selectbox("Acción", ["inicio", "fin"], key="rr_accion_res")
    with c5:
        analyst = st.text_input("Analista", key="rr_analista_res")
    with c6:
        st.write("")
        if st.button("Registrar Hora", type="primary", use_container_width=True, key="btn_registrar_res"):
            if selected_id:
                register_stage_time_for_sheet(selected_id, stage_action, action_type, analyst, sample_type, analisis)
                st.success("Registro aplicado a las muestras de la hoja seleccionada que coincidan con el tipo y análisis.")

    st.markdown("")
    section_header("% Cumplimiento global y Totales", CLR_RESUMEN)
    c1, c2, c3, c4 = st.columns(4)
    comp_df = compute_compliance_by_stage(sheets)
    comp_general = compliance_general(comp_df)
    with c1:
        st.markdown("**% Cumplimiento**")
        st.plotly_chart(donut(comp_general), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("**SLA**")
        st.markdown(f"<div style='font-size:32px;font-weight:700'>{SHEET_DEADLINE_LABEL}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Hojas**")
        st.markdown(f"<div style='font-size:32px;font-weight:700'>{len(sheets)}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("**Muestras**")
        total_samples = sum(len(sh.samples) for sh in sheets)
        st.markdown(f"<div style='font-size:32px;font-weight:700'>{total_samples}</div>", unsafe_allow_html=True)

    st.markdown("")
    section_header("Resumen de hojas de trabajo", CLR_RESUMEN)
    filter_date = st.date_input("Filtrar por fecha", value=None, format="YYYY-MM-DD", key="filter_date_resumen")
    view_sheets = [sh for sh in sheets if (not filter_date or sh.dateKey == filter_date.strftime("%Y-%m-%d"))]

    for sh in view_sheets:
        with st.expander(f"{sh.name} — Tipo: {sh.type} — {len(sh.samples)} muestras — Progreso {int(round(sheet_progress(sh)))}%"):
            st.caption(f"Creación: {sh.createdAt.strftime('%d-%m-%Y %H:%M:%S')} — Deadline {SHEET_DEADLINE_LABEL}")
            df = pd.DataFrame([{
                "ID": s.id,
                "Nombre": s.name,
                "Tipo": s.type,
                "Análisis": s.analisis,
                "Analista": s.analyst,
                "Progreso (%)": round(sample_progress(s.stages), 0),
                "Tiempo transcurrido": get_elapsed_str(s.addedAt),
                "Deadline": SHEET_DEADLINE_LABEL
            } for s in sh.samples])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ========= TAB 1: KPI =========
with tabs[1]:
    section_header("KPI - Cumplimiento y Duraciones", CLR_KPI)
    comp_df = compute_compliance_by_stage(sheets)
    comp_general = compliance_general(comp_df)

    k1, k2 = st.columns([1, 2])
    with k1:
        st.markdown("### Cumplimiento General")
        st.plotly_chart(donut(comp_general), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"SLA global: {SHEET_DEADLINE_LABEL}")
    with k2:
        st.markdown("### Resumen Operacional")
        total_samples = sum(len(sh.samples) for sh in sheets)
        tat_min = average_turnaround_minutes(sheets)
        tat_str = minutes_to_hhmm(tat_min) if np.isfinite(tat_min) else "N/A"
        comp_count = count_completed_validation(sheets)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hojas", len(sheets))
        m2.metric("Muestras", total_samples)
        m3.metric("T. resp. promedio (hasta Validación)", tat_str, help=f"{comp_count} muestras completas")
        m4.metric("SLA", SHEET_DEADLINE_LABEL)

    st.markdown("---")
    st.markdown("### % Cumplimiento por Etapa")
    ring_cols = st.columns(6)
    for idx, row in enumerate(comp_df.itertuples(index=False)):
        with ring_cols[idx % 6]:
            st.plotly_chart(donut(row.value), use_container_width=True, config={"displayModeBar": False})
            st.caption(row.name)

    st.markdown("---")
    st.markdown("### Tiempo promedio por etapa (min) vs SLA (min)")
    df_dur = calculate_stage_durations(sheets)
    fig_bar1 = px.bar(df_dur, x="etapa", y=["promedio", "sla"], barmode="group",
                      labels={"value":"Minutos","variable":"Serie"}, height=360)
    st.plotly_chart(fig_bar1, use_container_width=True, config={"displayModeBar": False})
    st.caption("* El promedio se calcula con las marcas de inicio y fin registradas por etapa.")

    st.markdown("### Gap promedio entre etapas consecutivas (min y horas)")
    df_gap = calculate_interstage_gaps(sheets)
    fig_bar2 = px.bar(df_gap, x="transicion", y=["gapPromedio", "gapHoras"], barmode="group",
                      labels={"value":"Valor","variable":"Serie"}, height=380)
    st.plotly_chart(fig_bar2, use_container_width=True, config={"displayModeBar": False})
    st.caption("* Gap = tiempo desde el fin de la etapa i al inicio de la etapa i+1.")

    st.markdown("### Plazos base por etapa (min)")
    df_base = pd.DataFrame([{"etapa": k, "minutos": v} for k, v in STAGE_DEADLINES.items()])
    fig_bar3 = px.bar(df_base, x="etapa", y="minutos", labels={"minutos":"Minutos"} , height=300)
    st.plotly_chart(fig_bar3, use_container_width=True, config={"displayModeBar": False})
    st.caption("* Valores SLA de referencia.")

# ========= TAB 2: Hojas de trabajo (Registro rápido + listado) =========
with tabs[2]:
    section_header("Registro Rápido de Avance", CLR_HOJAS)
    opts = [(sh.id, sh.name) for sh in sheets]
    sel = st.selectbox("Hoja", options=opts, format_func=lambda x: x[1], key="selected_sheet_id")
    selected_id = sel[0] if isinstance(sel, tuple) else sel

    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
    with c1:
        sample_type = st.selectbox("Tipo de muestra", ["Metálico", "No Metálico"], key="rr_tipo")
    with c2:
        analisis = st.selectbox("Análisis", ANALISIS_BY_TYPE.get(sample_type, ["-"]), key="rr_an")
    with c3:
        stage_action = st.selectbox("Etapa", list(STAGE_DEADLINES.keys()), key="rr_etapa")
    with c4:
        action_type = st.selectbox("Acción", ["inicio", "fin"], key="rr_accion")
    with c5:
        analyst = st.text_input("Analista", key="rr_analista")
    with c6:
        st.write("")
        if st.button("Registrar Hora", type="primary", use_container_width=True, key="btn_registrar"):
            if selected_id:
                register_stage_time_for_sheet(selected_id, stage_action, action_type, analyst, sample_type, analisis)
                st.success("Registro aplicado a las muestras de la hoja seleccionada que coincidan con el tipo y análisis.")

    st.markdown("")
    section_header("Listado de Hojas de trabajo", CLR_HOJAS)
    for sh in sheets:
        with st.expander(f"{sh.name} — Tipo: {sh.type} — {len(sh.samples)} muestras — Progreso {int(round(sheet_progress(sh)))}%"):
            st.caption(f"Creación: {sh.createdAt.strftime('%d-%m-%Y %H:%M:%S')} — Deadline {SHEET_DEADLINE_LABEL}")
            df = pd.DataFrame([{
                "ID": s.id,
                "Nombre": s.name,
                "Tipo": s.type,
                "Análisis": s.analisis,
                "Analista": s.analyst,
                "Progreso (%)": round(sample_progress(s.stages), 0),
                "Tiempo transcurrido": get_elapsed_str(s.addedAt),
                "Deadline": SHEET_DEADLINE_LABEL
            } for s in sh.samples])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ========= TAB 3: Ingreso =========
with tabs[3]:
    section_header("Ingreso de Muestras (Nueva hoja de trabajo)", CLR_PANEL)
    # Tabla editable simple
    for idx, row in enumerate(st.session_state.bulk_rows):
        cols = st.columns([1, 2, 1.4, 1.8, 1.8, 0.8])
        with cols[0]:
            st.text_input("ID", key=f"bulk_id_{idx}", value=row["id"], on_change=lambda i=idx: st.session_state.bulk_rows.__setitem__(i, {**st.session_state.bulk_rows[i], "id": st.session_state[f'bulk_id_{i}']}))
        with cols[1]:
            st.text_input("Nombre/Descripción", key=f"bulk_name_{idx}", value=row["name"], on_change=lambda i=idx: st.session_state.bulk_rows.__setitem__(i, {**st.session_state.bulk_rows[i], "name": st.session_state[f'bulk_name_{i}']}))
        with cols[2]:
            t = st.selectbox("Tipo", ["Metálico", "No Metálico"], key=f"bulk_type_{idx}", index=(0 if row['type']=="Metálico" else 1))
            st.session_state.bulk_rows[idx]["type"] = t
        with cols[3]:
            an_list = ANALISIS_BY_TYPE.get(t, ["-"])
            a = st.selectbox("Análisis", an_list, key=f"bulk_an_{idx}", index=min(an_list.index(row["analisis"]) if row["analisis"] in an_list else 0, len(an_list)-1))
            st.session_state.bulk_rows[idx]["analisis"] = a
        with cols[4]:
            st.text_input("Analista", key=f"bulk_analyst_{idx}", value=row["analyst"], on_change=lambda i=idx: st.session_state.bulk_rows.__setitem__(i, {**st.session_state.bulk_rows[i], "analyst": st.session_state[f'bulk_analyst_{i}']}))
        with cols[5]:
            if st.button("Quitar", key=f"bulk_remove_{idx}"):
                remove_bulk_row(idx)
                st.experimental_rerun()

    st.write("")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("+ Agregar muestra", key="bulk_add"):
            add_bulk_row()
    with c2:
        if st.button("Guardar hoja", type="primary", key="bulk_save"):
            save_batch()
            st.success("Nueva hoja creada.")
            st.experimental_rerun()

    st.caption("* Se crea una nueva hoja nombrada por fecha y número del día (ej. 04-09-2025/2).")
