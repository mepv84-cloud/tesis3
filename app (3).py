
import streamlit as st
import math
from datetime import date

st.set_page_config(page_title="Lab Dashboard | Mockup sin dependencias", layout="wide")

# ======================== STYLES ========================
st.markdown(
    "<style>"
    ":root{--bg:#0b1220;--card:#111827;--muted:#6b7280;--fg:#e5e7eb;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#3b82f6;--ring:#1f2937;--border:#1f2937;}"
    "html,body,[data-testid='stAppViewContainer']{background:var(--bg);color:var(--fg);}"
    "h1,h2,h3,h4{color:var(--fg);}"
    ".card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 1px 0 rgba(255,255,255,0.03) inset,0 10px 30px rgba(0,0,0,0.25);}"
    ".kpi{display:flex;align-items:center;gap:14px;}"
    ".kpi .icon{width:40px;height:40px;border-radius:10px;display:grid;place-items:center;font-weight:700;background:#0b1530;color:#b3c7ff;border:1px solid #1d2a4d;}"
    ".kpi .label{color:var(--muted);font-size:13px;}"
    ".kpi .value{font-size:26px;font-weight:700;}"
    "hr{border:none;border-top:1px solid #1f2937;margin:18px 0;}"
    ".barRow{margin:10px 0;}"
    ".barBg{background:#1f2937;height:14px;border-radius:10px;overflow:hidden;}"
    ".barFg{height:100%;background:var(--green);}"
    ".table{width:100%;border-collapse:collapse;}"
    ".table th,.table td{padding:10px;border-bottom:1px solid #1f2937;}"
    ".table th{color:#9ca3af;font-weight:600;text-align:left;}"
    ".badge{padding:3px 8px;border-radius:999px;font-size:12px;border:1px solid #263143;background:#0f192f;}"
    "</style>",
    unsafe_allow_html=True
)

# ======================== DATA (mock) ========================
stages = ["Recepción", "Preparación", "Análisis", "Revisión", "Informe"]
done   = [92, 85, 78, 88, 95]
total  = [100, 100, 100, 100, 100]
late   = [t - d for t, d in zip(total, done)]
pct    = [ (d/t*100 if t else 0.0) for d, t in zip(done, total) ]

total_total = sum(total)
total_done  = sum(done)
total_late  = sum(late)
pct_general = (total_done/total_total*100) if total_total else 0.0

# ======================== HELPERS ========================
def donut_svg(percent, size=220, stroke=22, color_ok="#22c55e", color_no="#293142", label=""):
    r = (size - stroke) / 2
    cx = cy = size/2
    C = 2 * math.pi * r
    done_len = max(0, min(100, percent)) / 100.0 * C
    remain = C - done_len
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color_no}" stroke-width="{stroke}" fill="none" stroke-linecap="round" stroke-dasharray="{C} {C}" />'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color_ok}" stroke-width="{stroke}" fill="none" stroke-linecap="round" stroke-dasharray="{done_len} {remain}" transform="rotate(-90 {cx} {cy})" />'
        f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="{size*0.18:.0f}" font-family="ui-sans-serif,system-ui" fill="#e5e7eb">{percent:.1f}%</text>'
        f'<text x="50%" y="{size*0.92:.0f}" dominant-baseline="middle" text-anchor="middle" font-size="{size*0.10:.0f}" font-family="ui-sans-serif,system-ui" fill="#9ca3af">{label}</text>'
        f'</svg>'
    )

def bar_row(label, percent, color="#22c55e"):
    pct_str = f"{percent:.1f}%"
    return (
        f'<div class="barRow">'
        f'  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'    <span style="color:#cbd5e1">{label}</span>'
        f'    <span style="color:#cbd5e1;font-variant-numeric:tabular-nums">{pct_str}</span>'
        f'  </div>'
        f'  <div class="barBg"><div class="barFg" style="width:{percent}%; background:{color}"></div></div>'
        f'</div>'
    )

# ======================== HEADER / FILTERS ========================
left, mid, right = st.columns([1.3, 2, 1.1])
with left:
    st.markdown('<div class="card"><h2 style="margin:0">🔬 Laboratorio | SLA Dashboard</h2><div style="color:#9ca3af">Mockup visual similar al diseño original</div></div>', unsafe_allow_html=True)
with mid:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            _ = st.selectbox("Tipo de muestra", ["Todas","Metálicas","No metálicas"], index=0)
        with c2:
            _ = st.selectbox("Cliente", ["Todos","Cliente A","Cliente B"], index=0)
        with c3:
            _ = st.date_input("Rango de fechas", (date(2025,1,1), date(2025,12,31)))
with right:
    st.markdown('<div class="card" style="text-align:right;"><div style="color:#9ca3af;">Estado general</div>'
                f'<div style="font-size:28px;font-weight:700;color:#e5e7eb">{pct_general:.1f}%</div>'
                '<div><span class="badge">SLA mensual</span></div></div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)

# ======================== KPI CARDS ========================
k1, k2, k3, k4 = st.columns(4)
st.markdown('<div class="card kpi"><div class="icon">OK</div><div><div class="label">Cumplidas</div>'
            f'<div class="value">{total_done}</div></div></div>', unsafe_allow_html=True, help="Total de muestras dentro de SLA")
with k2:
    st.markdown('<div class="card kpi"><div class="icon">⏰</div><div><div class="label">Fuera de plazo</div>'
                f'<div class="value" style="color:#ef4444">{total_late}</div></div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="card kpi"><div class="icon">📦</div><div><div class="label">Total mensual</div>'
                f'<div class="value">{total_total}</div></div></div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="card kpi"><div class="icon">%</div><div><div class="label">% objetivo</div>'
                f'<div class="value" style="color:#22c55e">{pct_general:.1f}%</div></div></div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)

# ======================== MAIN ROW ========================
cA, cB = st.columns([1.1, 1.6], gap="large")
with cA:
    st.markdown('<div class="card"><h4>✅ Cumplimiento general</h4>' + donut_svg(pct_general, size=260, stroke=28, label="Cumplimiento") +
                '<hr/><div style="display:flex;gap:10px;color:#9ca3af">'
                f'<div>Totales: <b style="color:#e5e7eb">{total_total}</b></div>'
                f'<div>| Cumplidas: <b style="color:#22c55e">{total_done}</b></div>'
                f'<div>| No cumplidas: <b style="color:#ef4444">{total_late}</b></div>'
                '</div></div>', unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card"><h4>🧭 % cumplimiento por etapa</h4>' +
                "".join(bar_row(e, p) for e, p in sorted(zip(stages, pct), key=lambda x: x[1])) +
                '</div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)

# ======================== GRID RINGS ========================
st.markdown("#### 🧩 Anillos por etapa")
grid = st.columns(5)
for i, (e, p) in enumerate(zip(stages, pct)):
    with grid[i % 5]:
        st.markdown('<div class="card" style="text-align:center;">' + donut_svg(p, size=200, stroke=22, label=e) + '</div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)

# ======================== TABLE ========================
rows = "".join(f"<tr><td>{e}</td><td>{d}</td><td>{t}</td><td>{p:.1f}%</td><td><span class='badge'>OK</span></td></tr>"
               for e, d, t, p in zip(stages, done, total, pct))
st.markdown('<div class="card"><h4>📄 Actividad reciente</h4>'
            '<table class="table"><thead><tr><th>Etapa</th><th>Cumplidas</th><th>Total</th><th>%</th><th>Estado</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
