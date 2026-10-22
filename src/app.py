"""Streamlit web interface for the Civic & Municipality Complaint Classification System.

A clean, light admin-dashboard style interface (white sidebar with grouped
sections, green accent, top bar, stat cards, focused content pages). It loads
the *saved* final preprocessing-and-model pipelines and never retrains.

Run from the project root:
    streamlit run src/app.py
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

# Allow "streamlit run src/app.py" by making the package importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.prepare import (  # noqa: E402
    PROJECT_ROOT,
    TARGET_SERVICE,
    TARGET_URGENCY,
    audit_summary,
)
from src.predict import (  # noqa: E402
    available_targets,
    build_input_frame,
    load_meta,
    load_pipeline,
    predict,
    validate_complaint,
)

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")

TARGET_TITLES = {TARGET_SERVICE: "Service Category", TARGET_URGENCY: "Urgency Level"}

SERVICE_PRETTY = {
    "water_leakage": "Water Leakage", "drainage": "Drainage",
    "pothole": "Pothole / Road", "streetlight": "Streetlight",
    "garbage": "Garbage / Waste", "noise": "Noise",
    "illegal_parking": "Illegal Parking", "graffiti": "Graffiti", "other": "Other",
}
URGENCY_PRETTY = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
URGENCY_COLOR = {"low": "#16a34a", "medium": "#d97706", "high": "#ea580c", "critical": "#dc2626"}
SERVICE_ICON = {
    "water_leakage": "💧", "drainage": "🌧️", "pothole": "🛣️", "streetlight": "💡",
    "garbage": "🗑️", "noise": "🔊", "illegal_parking": "🚗", "graffiti": "🎨", "other": "📋",
}


def pretty(target: str, label: str) -> str:
    if target == TARGET_SERVICE:
        return SERVICE_PRETTY.get(label, label.replace("_", " ").title())
    return URGENCY_PRETTY.get(label, label.title())


# --------------------------------------------------------------------------- #
# Page config + styling (light admin dashboard, green accent)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="CityDesk — Municipal Complaints",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --accent:#16a34a; --accent-soft:#e7f6ec; --accent-dark:#15803d;
  --ink:#111827; --muted:#6b7280; --line:#eceef1; --bg:#ffffff; --panel:#f8f9fb;
}
html, body, [class*="css"], .stApp {
  font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);
}
.stApp { background:var(--bg); }
.block-container { padding-top:1.1rem; padding-bottom:3rem; max-width:1250px; }
h1,h2,h3,h4 { color:var(--ink); font-weight:700; letter-spacing:-0.02em; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; }

/* ---------- Sidebar: white, grouped ---------- */
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
.sb-brand { display:flex; align-items:center; gap:0.55rem; padding:0.2rem 0.3rem 0.9rem; }
.sb-logo { width:34px; height:34px; border-radius:9px;
  background:var(--accent); display:flex; align-items:center; justify-content:center;
  font-size:1.1rem; }
.sb-name { font-size:1.15rem; font-weight:800; color:var(--ink); }
.sb-name .g { color:var(--accent); }
.sb-group { font-size:0.66rem; letter-spacing:0.1em; text-transform:uppercase;
  color:#9aa3af; margin:1.1rem 0 0.2rem 0.5rem; font-weight:700; }

/* radio -> nav rows */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:0.12rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  border-radius:9px; padding:0.5rem 0.7rem; margin:0; cursor:pointer; width:100%;
  color:#374151; font-weight:500; font-size:0.92rem; transition:all .12s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:var(--panel); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background:var(--accent-soft); color:var(--accent-dark); font-weight:700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input { display:none; }

.sb-profile { display:flex; align-items:center; gap:0.55rem; margin-top:1rem;
  padding:0.6rem; border:1px solid var(--line); border-radius:11px; background:var(--panel); }
.sb-avatar { width:30px; height:30px; border-radius:50%; background:var(--accent);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.85rem; }
.sb-pname { font-size:0.88rem; font-weight:700; }
.sb-prole { font-size:0.72rem; color:var(--muted); }

/* ---------- Top bar ---------- */
.topbar-date { font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;
  color:var(--muted); font-weight:700; }
.topbar-title { font-size:1.9rem; font-weight:800; margin:0.1rem 0 0; }
.topbar-sub { color:var(--muted); font-size:0.9rem; margin-top:0.15rem; }

/* ---------- Stat cards ---------- */
.stat { background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:1rem 1.1rem; box-shadow:0 1px 2px rgba(16,24,40,0.03); height:100%; }
.stat .ic { width:34px; height:34px; border-radius:9px; background:var(--panel);
  display:flex; align-items:center; justify-content:center; font-size:1.05rem; }
.stat .v { font-size:1.85rem; font-weight:800; margin-top:0.55rem; line-height:1; }
.stat .l { font-size:0.68rem; letter-spacing:0.07em; text-transform:uppercase;
  color:var(--muted); margin-top:0.35rem; font-weight:700; }
.stat .v.red { color:#dc2626; }

/* ---------- Panels ---------- */
.panel { background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:1.1rem 1.25rem; box-shadow:0 1px 2px rgba(16,24,40,0.03); margin-bottom:1rem; }
.panel-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:0.7rem; }
.panel-title { font-size:1rem; font-weight:700; display:flex; align-items:center; gap:0.4rem; }
.panel-link { color:var(--accent); font-size:0.82rem; font-weight:600; }

.scorebar { border:1px solid var(--line); border-radius:11px; overflow:hidden; }
.scorebar .top { background:var(--panel); padding:0.5rem 0.8rem; font-size:0.8rem;
  font-weight:700; display:flex; justify-content:space-between; }
.scorebar .row { display:flex; justify-content:space-between; padding:0.55rem 0.9rem;
  border-top:1px solid var(--line); font-weight:600; }
.scorebar .foot { padding:0.45rem 0.9rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:0.76rem; }
.livetag { color:#dc2626; font-size:0.7rem; font-weight:800; }

/* result card */
.result { border-radius:14px; padding:1.2rem 1.3rem; color:#fff; margin-bottom:0.5rem; }
.result .lab { font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.92; }
.result .big { font-size:1.7rem; font-weight:800; margin:0.2rem 0; }
.result .mdl { font-size:0.78rem; opacity:0.92; }

.notice { background:#fffbeb; border:1px solid #fde68a; border-radius:12px;
  padding:0.85rem 1.05rem; color:#92610a; font-size:0.86rem; }
.subtle { color:var(--muted); font-size:0.88rem; }

/* main-content buttons = green pill */
.stButton>button, .stFormSubmitButton>button {
  background:var(--accent); color:#fff; border:none; border-radius:999px;
  padding:0.5rem 1.2rem; font-weight:700; font-size:0.9rem; }
.stButton>button:hover, .stFormSubmitButton>button:hover { background:var(--accent-dark); color:#fff; }

/* sidebar nav buttons = left-aligned rows, not pills */
section[data-testid="stSidebar"] .stButton>button {
  border:none; border-radius:9px; padding:0.5rem 0.7rem; font-weight:500;
  font-size:0.92rem; text-align:left; justify-content:flex-start; box-shadow:none; }
/* inactive (secondary) nav row */
section[data-testid="stSidebar"] .stButton>button[kind="secondary"] {
  background:transparent; color:#374151; }
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {
  background:var(--panel); color:var(--ink); }
/* active (primary) nav row = soft green */
section[data-testid="stSidebar"] .stButton>button[kind="primary"] {
  background:var(--accent-soft); color:var(--accent-dark); font-weight:700; }
section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover {
  background:#d8f0df; color:var(--accent-dark); }
section[data-testid="stSidebar"] .stButton>button p { text-align:left; width:100%; margin:0; }
.footer { color:#9aa3af; font-size:0.76rem; text-align:center; margin-top:2rem;
  padding-top:1rem; border-top:1px solid var(--line); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_pipeline(target: str):
    return load_pipeline(target)


@st.cache_data(show_spinner=False)
def get_meta(target: str):
    return load_meta(target)


@st.cache_data(show_spinner=False)
def get_audit():
    return audit_summary()


@st.cache_data(show_spinner=False)
def get_table(name: str):
    path = os.path.join(TABLES_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fig(name: str):
    path = os.path.join(FIG_DIR, name)
    return path if os.path.exists(path) else None


def goto(page: str):
    st.session_state["nav"] = page
    st.rerun()


def topbar(title: str, subtitle: str = "", actions=None):
    today = _dt.date.today().strftime("%A, %d %B %Y").upper()
    left, right = st.columns([3, 1.3])
    with left:
        st.markdown(f'<div class="topbar-date">{today}</div>'
                    f'<div class="topbar-title">{title}</div>'
                    + (f'<div class="topbar-sub">{subtitle}</div>' if subtitle else ""),
                    unsafe_allow_html=True)
    with right:
        if actions:
            cols = st.columns(len(actions))
            for c, (label, target) in zip(cols, actions):
                if c.button(label, key=f"act_{label}"):
                    goto(target)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


def stat(col, icon, value, label, red=False):
    vcls = "v red" if red else "v"
    col.markdown(f'<div class="stat"><div class="ic">{icon}</div>'
                 f'<div class="{vcls}">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def panel_head(title_html: str, link_label=None, link_target=None):
    link = ""
    if link_label:
        link = f'<span class="panel-link">{link_label} →</span>'
    st.markdown(f'<div class="panel-head"><div class="panel-title">{title_html}</div>{link}</div>',
                unsafe_allow_html=True)


def responsible_notice():
    st.markdown('<div class="notice">⚠️ <b>Advisory use only.</b> Predictions are '
                "decision-support suggestions and may be wrong. Staff must review them; "
                "they must not deny service or replace emergency procedures.</div>",
                unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer">CityDesk · Municipal Complaint Classifier · '
                "IDA Project · Predictions are advisory.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pages (kept focused — the dashboard only links out to detail pages)
# --------------------------------------------------------------------------- #
def page_dashboard():
    topbar("Dashboard", "Municipal complaint intelligence overview",
           actions=[("＋ Classify", "Classify Complaint"), ("📊 Explore", "Dataset Explorer")])

    audit = get_audit()
    svc = pd.Series(audit["service_distribution"])
    urg = pd.Series(audit["urgency_distribution"])
    high = int(urg.get("high", 0) + urg.get("critical", 0))

    c = st.columns(6)
    stat(c[0], "🗂️", f"{audit['n_rows_clean']:,}", "Records")
    stat(c[1], "🏷️", len(svc), "Categories")
    stat(c[2], "⚡", len(urg), "Urgency levels")
    stat(c[3], "🔴", f"{high:,}", "High + critical", red=True)
    stat(c[4], "🧮", len(audit["feature_columns"]), "Features")
    stat(c[5], "🤖", f"{len(available_targets())}/2", "Models live")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_head("🏷️ Complaints by category", "See all", "Dataset Explorer")
        s = svc.sort_values(ascending=False)
        s.index = [pretty(TARGET_SERVICE, i) for i in s.index]
        st.bar_chart(s, color="#16a34a", height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_head("🏆 Selected models", "Details", "Model Performance")
        for tgt in (TARGET_SERVICE, TARGET_URGENCY):
            meta = get_meta(tgt)
            metrics = get_table(f"test_metrics_{tgt}.csv")
            if meta and metrics is not None:
                row = metrics[metrics["model"] == meta["champion_model"]]
                f1 = float(row["test_macro_f1"].iloc[0]) if not row.empty else float("nan")
                st.markdown(
                    f'<div class="scorebar" style="margin-bottom:0.6rem">'
                    f'<div class="top"><span>{TARGET_TITLES[tgt]}</span>'
                    f'<span style="color:#16a34a">macro-F1 {f1:.3f}</span></div>'
                    f'<div class="row"><span>{meta["champion_label"]}</span>'
                    f'<span class="subtle">champion</span></div></div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    responsible_notice()
    footer()


def page_classify():
    topbar("Classify a Complaint",
           "Enter the complaint and approved intake details to get a suggestion.")
    targets = available_targets()
    if not targets:
        st.error("No trained pipeline found. Run `python -m src.train` then "
                 "`python -m src.optimise` first.")
        footer(); return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    with st.form("classify"):
        st.markdown("**Complaint description** *")
        complaint = st.text_area(
            "c", height=130, label_visibility="collapsed",
            placeholder="e.g. Main water pipe has burst and flooding is occurring near the school.",
            help="Plain language, minimum 12 characters.")
        a, b, cc = st.columns(3)
        city = a.text_input("City", value="New York")
        state = b.text_input("State", value="NY", help="e.g. NY, CA, TX")
        anon = cc.checkbox("Anonymous", value=False)
        d, e = st.columns(2)
        area = d.slider("Area importance", 1, 10, 7)
        reports = e.number_input("Related citizen reports", 1, 100, 20)
        submitted = st.form_submit_button("🔎  Classify complaint")
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        st.caption("Examples: “Streetlight has not been working for several days.” · "
                   "“Garbage collection was missed this week.”")
        footer(); return

    valid = validate_complaint(complaint)
    if not valid.ok:
        st.error("🚫 " + valid.message)
        footer(); return

    X = build_input_frame(complaint_text=complaint, location_city=city,
                          location_state=state, is_anonymous=anon,
                          area_importance=area, citizen_reports_count=reports)

    cols = st.columns(len(targets))
    for col, tgt in zip(cols, targets):
        result = predict(get_pipeline(tgt), X)
        meta = get_meta(tgt)
        label = result["label"]
        if tgt == TARGET_SERVICE:
            bg = "linear-gradient(135deg,#16a34a,#15803d)"; icon = SERVICE_ICON.get(label, "📋")
        else:
            cc = URGENCY_COLOR.get(label, "#16a34a"); bg = f"linear-gradient(135deg,{cc},{cc})"; icon = "⚡"
        with col:
            st.markdown(f'<div class="result" style="background:{bg}">'
                        f'<div class="lab">{TARGET_TITLES[tgt]}</div>'
                        f'<div class="big">{icon} {pretty(tgt, label)}</div>'
                        f'<div class="mdl">Model: {meta["champion_label"] if meta else "—"}</div></div>',
                        unsafe_allow_html=True)
            if result["probabilities"]:
                top = list(result["probabilities"].items())[:5]
                df = pd.DataFrame([(pretty(tgt, k), round(v, 4)) for k, v in top],
                                  columns=["class", "probability"]).set_index("class")
                st.bar_chart(df, color="#16a34a", height=210)
                st.caption(f"Top confidence: {top[0][1]:.0%} (may not be calibrated).")

    st.markdown('<div class="panel">🧑‍⚖️ <b>Staff review required.</b> This is a '
                "suggestion and does not route, close, or deny any complaint "
                "automatically.</div>", unsafe_allow_html=True)
    responsible_notice()
    footer()


def page_dataset():
    topbar("Dataset Explorer", "Charts derived directly from the cleaned dataset.")
    audit = get_audit()
    c = st.columns(4)
    stat(c[0], "🗂️", f"{audit['n_rows_clean']:,}", "Cleaned records")
    stat(c[1], "🧹", audit["duplicate_report_ids"], "Duplicates removed")
    stat(c[2], "🏷️", len(audit["service_distribution"]), "Categories")
    stat(c[3], "⚡", len(audit["urgency_distribution"]), "Urgency levels")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    pairs = [
        ("service_distribution.png", "Service categories (moderately imbalanced)."),
        ("urgency_distribution.png", "Urgency levels (medium is most common)."),
        ("service_vs_urgency.png", "Category vs urgency — some categories skew urgent."),
        ("missingness.png", "Missing values concentrate in excluded post-outcome / PII fields."),
        ("complaints_over_time.png", "Monthly complaint volume."),
        ("text_length.png", "Complaint text-length distribution."),
    ]
    for i in range(0, len(pairs), 2):
        cols = st.columns(2)
        for col, (img, cap) in zip(cols, pairs[i:i + 2]):
            if fig(img):
                with col:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.image(fig(img), use_container_width=True)
                    st.caption(cap)
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><b>Privacy & leakage controls</b><br>'
                '<span class="subtle">Leakage columns: '
                + ", ".join(f"<code>{c}</code>" for c in audit["leakage_columns"])
                + "<br>PII excluded (POPIA): "
                + ", ".join(f"<code>{c}</code>" for c in audit["pii_columns"])
                + "</span></div>", unsafe_allow_html=True)
    footer()


def _metrics_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["label", "test_accuracy", "test_macro_f1", "test_weighted_f1",
            "test_macro_precision", "test_macro_recall", "roc_auc_ovr_macro", "pr_auc_macro"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    rename = {"label": "Model", "test_accuracy": "Accuracy", "test_macro_f1": "Macro-F1",
              "test_weighted_f1": "Weighted-F1", "test_macro_precision": "Precision",
              "test_macro_recall": "Recall", "roc_auc_ovr_macro": "ROC-AUC", "pr_auc_macro": "PR-AUC"}
    for c in out.columns:
        if c != "label":
            out[c] = out[c].round(3)
    return out.rename(columns=rename)


def page_performance():
    topbar("Model Performance", "Measured on the untouched 20% held-out test set.")
    tgt = st.selectbox("Task", [TARGET_SERVICE, TARGET_URGENCY],
                       format_func=lambda t: TARGET_TITLES[t])
    metrics = get_table(f"test_metrics_{tgt}.csv")
    if metrics is None:
        st.info("Run `python -m src.evaluate` first."); footer(); return

    meta = get_meta(tgt)
    best = metrics.loc[metrics["test_macro_f1"].idxmax()]
    c = st.columns(3)
    stat(c[0], "🏆", meta["champion_label"] if meta else "—", "Selected model")
    stat(c[1], "🎯", f"{best['test_macro_f1']:.3f}", "Best macro-F1")
    stat(c[2], "✅", f"{best['test_accuracy']:.3f}", "Best accuracy")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_head("📈 All models — test metrics")
    st.dataframe(_metrics_display(metrics), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_head("🔲 Confusion matrices")
    names = metrics["model"].tolist()
    default = [m for m in ["baseline", meta["champion_model"]] if meta and m in names] or names[:2]
    chosen = st.multiselect("Models", names, default=default)
    cols = st.columns(2)
    for i, m in enumerate(chosen):
        f = fig(f"confusion_{tgt}_{m}.png")
        if f:
            cols[i % 2].image(f, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_head("📉 ROC & Precision-Recall curves")
    a, b = st.columns(2)
    if fig(f"roc_{tgt}.png"):
        a.image(fig(f"roc_{tgt}.png"), use_container_width=True)
    if fig(f"pr_{tgt}.png"):
        b.image(fig(f"pr_{tgt}.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    footer()


def page_comparison():
    topbar("Model Comparison", "Cross-validation, optimisation, ablations, McNemar's test.")
    tgt = st.selectbox("Task", [TARGET_SERVICE, TARGET_URGENCY],
                       format_func=lambda t: TARGET_TITLES[t], key="cmp")

    cv = get_table(f"cv_{tgt}.csv")
    if cv is not None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_head("🔁 Cross-validation (5-fold, training set)")
        show = cv[["label", "val_macro_f1_mean", "val_macro_f1_sd",
                   "val_accuracy_mean", "overfit_gap_macro_f1"]].copy()
        show.columns = ["Model", "Val Macro-F1", "Val Macro-F1 SD", "Val Accuracy", "Overfit gap"]
        for c in show.columns[1:]:
            show[c] = show[c].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    pp = get_table(f"pre_post_{tgt}.csv")
    ab = get_table(f"ablations_{tgt}.csv")
    cc1, cc2 = st.columns(2)
    if pp is not None:
        with cc1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_head("🛠️ Before vs after optimisation")
            p = pp.copy()
            for c in p.columns:
                if c != "config":
                    p[c] = p[c].round(3)
            st.dataframe(p, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
    if ab is not None:
        with cc2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            panel_head("🧪 Ablations (one change each)")
            a = ab.copy()
            for c in a.columns:
                if c != "ablation":
                    a[c] = a[c].round(3)
            st.dataframe(a, use_container_width=True, hide_index=True)
            st.caption("A=remove text · B=remove location · C=remove class weighting.")
            st.markdown("</div>", unsafe_allow_html=True)

    mc = get_table(f"mcnemar_{tgt}.csv")
    if mc is not None:
        row = mc.iloc[0].to_dict()
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_head("📐 McNemar's test (champion vs comparator)")
        c = st.columns(3)
        stat(c[0], "🔀", int(row.get("n_discordant", 0)), "Discordant pairs")
        stat(c[1], "📏", f"{float(row.get('p_value', float('nan'))):.4f}", "p-value")
        stat(c[2], "📊", "Yes" if bool(row.get("significant_at_0.05")) else "No", "Significant α=0.05")
        st.caption(f"Champion: {row.get('champion')} · comparator: {row.get('comparator')} · "
                   f"method: {row.get('method')}.")
        st.markdown("</div>", unsafe_allow_html=True)
    footer()


def page_about():
    topbar("About", "Civic & Municipality Complaint Classification System.")
    st.markdown('<div class="panel"><b>Pipeline</b><br><span class="subtle">'
                "cleaning & canonicalisation → leakage and PII removal → TF-IDF text "
                "+ one-hot categorical + scaled numeric features → five models "
                "(Dummy baseline, Logistic Regression, Random Forest, MLP, Tabular "
                "Neural Network) → 5-fold CV, held-out test evaluation, RandomizedSearch "
                "optimisation, ablations and McNemar's test.</span></div>",
                unsafe_allow_html=True)
    m_s, m_u = get_meta(TARGET_SERVICE), get_meta(TARGET_URGENCY)
    c = st.columns(2)
    if m_s:
        stat(c[0], "🏷️", m_s["champion_label"], "Service-category model")
    if m_u:
        stat(c[1], "⚡", m_u["champion_label"], "Urgency model")
    st.markdown('<div class="panel"><b>Responsible use & privacy</b><br>'
                '<span class="subtle">'
                "• Predictions are advisory and may be wrong; staff must review them.<br>"
                "• The model must not auto-reject complaints or deny services.<br>"
                "• Emergency procedures must not be replaced by predictions.<br>"
                "• Personal identifiers are excluded from training, charts, and logs."
                "</span></div>", unsafe_allow_html=True)
    responsible_notice()
    footer()


# --------------------------------------------------------------------------- #
# Navigation: grouped like the reference admin dashboard
# --------------------------------------------------------------------------- #
PAGES = {
    "Dashboard": ("🏠", page_dashboard),
    "Classify Complaint": ("📝", page_classify),
    "Dataset Explorer": ("📊", page_dataset),
    "Model Performance": ("📈", page_performance),
    "Model Comparison": ("⚖️", page_comparison),
    "About": ("ℹ️", page_about),
}
NAV_GROUPS = [
    ("Overview", ["Dashboard"]),
    ("Operations", ["Classify Complaint", "Dataset Explorer"]),
    ("Analytics", ["Model Performance", "Model Comparison"]),
    ("Info", ["About"]),
]


def main():
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"

    with st.sidebar:
        st.markdown('<div class="sb-brand"><div class="sb-logo">🏛️</div>'
                    '<div class="sb-name">City<span class="g">Desk</span></div></div>',
                    unsafe_allow_html=True)

        for group_name, items in NAV_GROUPS:
            st.markdown(f'<div class="sb-group">{group_name}</div>', unsafe_allow_html=True)
            for name in items:
                icon = PAGES[name][0]
                is_active = st.session_state["nav"] == name
                if st.button(f"{icon}  {name}", key=f"nav_{name}",
                             use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state["nav"] = name
                    st.rerun()

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        ready = len(available_targets())
        st.markdown(f'<div class="subtle" style="padding:0 0.5rem 0.4rem;font-size:0.76rem">'
                    f'🟢 {ready}/2 models loaded · saved pipeline</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sb-profile"><div class="sb-avatar">A</div>'
                    '<div><div class="sb-pname">Admin</div>'
                    '<div class="sb-prole">Triage supervisor</div></div></div>',
                    unsafe_allow_html=True)

    PAGES[st.session_state["nav"]][1]()


main()
