"""Render one static HTML file with embedded chart data and a shared Plotly CDN."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import ARTIFACTS_DIR, DOCS_DIR, OFFERS

TEAL = "#087b69"
VIOLET = "#6550bf"
AMBER = "#de9a41"
SLATE = "#536679"
INK = "#1d303b"
MODEL_COLORS = {"LogisticRegression": VIOLET, "HistGradientBoosting": TEAL}
RFM_NAMES = {
    "At_Risk": "В группе риска",
    "Champions": "Наиболее активные",
    "Engaged": "Вовлечённые",
    "Occasional": "Нерегулярные",
}
OFFER_NAMES = ("Без оффера", "Безлимит на 30 дней", "Скидка 20%", "Пакет роуминга")
STRATEGIES = {"do_nothing": "Без действий", "mass_discount": "Массовая скидка", "ml_nbo": "ML NBO"}


def segment_chart(rows: list[dict], names: dict | None = None) -> go.Figure:
    ordered = sorted(rows, key=lambda row: row["churn_rate"], reverse=True)
    labels = [(names or {}).get(row["segment"], row["segment"]) for row in ordered]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[row["churn_rate"] for row in ordered],
            marker_color=TEAL,
            customdata=[[row["observations"], row["avg_arpu_kzt"]] for row in ordered],
            hovertemplate=(
                "%{x}<br>Отток %{y:.1%}<br>Наблюдения %{customdata[0]:,}"
                "<br>Средний ARPU %{customdata[1]:,.0f} ₸<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def traffic_chart(mart: pd.DataFrame) -> go.Figure:
    observed = mart.loc[mart["has_previous_month"]]
    edges = np.arange(-100, 111, 10)
    labels = [f"{low:+d}–{high:+d}%" for low, high in zip(edges[:-1], edges[1:], strict=True)]
    labels[-1] = "≥ +100%"
    fig = go.Figure()
    for churn, name, color in ((0, "Остались", "#67baaa"), (1, "Ушли", VIOLET)):
        changes = observed.loc[observed["churn"].eq(churn), "data_gb_mom_change"]
        counts, _ = np.histogram(changes.clip(edges[0], edges[-1]), bins=edges)
        fig.add_bar(
            x=labels,
            y=(counts / max(len(changes), 1)).tolist(),
            name=name,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:.1%} наблюдений<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Изменение трафика к предыдущему месяцу")
    fig.update_yaxes(title="Доля внутри группы", tickformat=".0%")
    return fig


def quality_curve(metrics: dict, kind: str) -> go.Figure:
    axes = {
        "roc": ("fpr", "tpr", "False Positive Rate", "True Positive Rate"),
        "pr": ("recall", "precision", "Recall", "Precision"),
        "calibration": (
            "mean_predicted_probability",
            "fraction_positive",
            "Средний прогноз",
            "Фактическая доля оттока",
        ),
    }
    x_key, y_key, x_title, y_title = axes[kind]
    fig = go.Figure()
    baseline = [metrics["test_churn_rate"]] * 2 if kind == "pr" else [0, 1]
    fig.add_scatter(
        x=[0, 1],
        y=baseline,
        mode="lines",
        line={"color": SLATE, "dash": "dot", "width": 1.5},
        showlegend=False,
        hoverinfo="skip",
    )
    for name, scores in metrics["models"].items():
        curve = scores[f"{kind}_curve"]
        x, y = curve[x_key], curve[y_key]
        indexes = np.linspace(0, len(x) - 1, min(len(x), 1000), dtype=int)
        fig.add_scatter(
            x=[x[index] for index in indexes],
            y=[y[index] for index in indexes],
            name=name,
            mode="lines+markers" if kind == "calibration" else "lines",
            line={"color": MODEL_COLORS[name], "width": 2.5},
            hovertemplate="%{x:.1%} / %{y:.1%}<extra>%{fullData.name}</extra>",
        )
    fig.update_xaxes(title=x_title, tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title=y_title, tickformat=".0%", range=[0, 1])
    return fig


def importance_chart(metrics: dict) -> go.Figure:
    rows = list(reversed(metrics["permutation_importance"]))
    fig = go.Figure(
        go.Bar(
            y=[row["feature"] for row in rows],
            x=[row["mean_auc_drop"] for row in rows],
            error_x={"type": "data", "array": [row["std_auc_drop"] for row in rows]},
            marker_color=TEAL,
            orientation="h",
            hovertemplate="%{y}<br>Δ ROC-AUC %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Падение ROC-AUC после перемешивания")
    return fig


def policy_chart(simulation: dict) -> go.Figure:
    rows = [simulation["strategies"][key] for key in STRATEGIES]
    fig = go.Figure()
    for key, label, color in (
        ("net_retained_margin_kzt", "Чистая маржа", TEAL),
        ("campaign_cost_kzt", "Затраты кампании", AMBER),
    ):
        fig.add_bar(
            x=list(STRATEGIES.values()),
            y=[row[key] / 1_000_000 for row in rows],
            name=label,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:,.1f} млн ₸<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="млн ₸", rangemode="tozero")
    return fig


def offer_chart(simulation: dict) -> go.Figure:
    rows = simulation["ml_offer_counts_by_rfm"]
    fig = go.Figure()
    colors = ("#aab8c5", TEAL, AMBER, VIOLET)
    for offer, label, color in zip(OFFERS, OFFER_NAMES, colors, strict=True):
        fig.add_bar(
            x=[RFM_NAMES[row["rfm_segment"]] for row in rows],
            y=[row[offer] for row in rows],
            name=label,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:,} абонентов<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Абоненты", tickformat="~s")
    return fig


def format_value(value: float, precision: int = 1) -> str:
    return f"{value:,.{precision}f}".replace(",", " ").replace(".", ",")


def chart_card(chart_id: str, title: str, description: str, wide: bool = False) -> str:
    classes = "card wide" if wide else "card"
    return (
        f'<article class="{classes}"><h3 id="title-{chart_id}">{escape(title)}</h3>'
        f'<p>{escape(description)}</p><div class="plot" id="{chart_id}" '
        f'aria-labelledby="title-{chart_id}"></div></article>'
    )


def strategy_table(simulation: dict) -> str:
    rows = []
    for key, label in STRATEGIES.items():
        row = simulation["strategies"][key]
        roi = "—" if row["roi_pct"] is None else f"{format_value(row['roi_pct'])}%"
        rows.append(
            f'<tr><th scope="row">{label}</th><td>{format_value(100 * row["churn_rate"], 2)}%</td>'
            f"<td>{format_value(row['lost_revenue_kzt'] / 1e6, 2)} млн ₸</td>"
            f"<td>{format_value(row['campaign_cost_kzt'] / 1e6, 2)} млн ₸</td>"
            f"<td>{format_value(row['net_incremental_margin_kzt'] / 1e6, 2)} млн ₸</td>"
            f"<td>{roi}</td></tr>"
        )
    lo, hi = simulation["strategies"]["ml_nbo"]["net_incremental_margin_ci95_kzt"]
    interval = f"{format_value(lo / 1e6, 2)} … {format_value(hi / 1e6, 2)} млн ₸"
    return f"""<article class="card wide"><h3>Итоги удержания</h3>
      <p>Интервал 95% для чистого эффекта ML NBO: {interval}.</p>
      <div class="table-wrap" tabindex="0" aria-label="Сравнение стратегий удержания"><table>
      <thead><tr><th scope="col">Стратегия</th><th scope="col">Отток</th>
      <th scope="col">Потерянная выручка</th><th scope="col">Затраты</th>
      <th scope="col">Эффект к контролю</th><th scope="col">ROI</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table></div></article>"""


def build_dashboard(
    mart: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
    output: Path = DOCS_DIR / "index.html",
) -> Path:
    eda, metrics, simulation = [
        json.loads((artifacts_dir / filename).read_text("utf-8"))
        for filename in ("eda_summary.json", "model_metrics.json", "nbo_simulation.json")
    ]
    winner = metrics["models"][metrics["best_model"]]
    nbo = simulation["strategies"]["ml_nbo"]
    roi = "—" if nbo["roi_pct"] is None else f"{format_value(nbo['roi_pct'])}%"
    indicators = [
        ("Абоненты", format_value(simulation["subscribers"], 0)),
        ("Наблюдаемый отток", f"{format_value(100 * metrics['test_churn_rate'], 2)}%"),
        ("Test ROC-AUC", format_value(winner["roc_auc"], 3)),
        ("Lift@10%", f"{format_value(winner['lift_at_10_pct'], 2)}×"),
        ("Чистый эффект NBO", f"{format_value(nbo['net_incremental_margin_kzt'] / 1e6, 2)} млн ₸"),
        ("ROI кампании", roi),
    ]
    cards = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in indicators)
    charts = {
        "tariffs": segment_chart(eda["tariff_plan"]),
        "rfm": segment_chart(eda["rfm_segment"], RFM_NAMES),
        "traffic": traffic_chart(mart),
        "roc": quality_curve(metrics, "roc"),
        "pr": quality_curve(metrics, "pr"),
        "calibration": quality_curve(metrics, "calibration"),
        "importance": importance_chart(metrics),
        "policies": policy_chart(simulation),
        "offers": offer_chart(simulation),
    }
    shares_legend = {
        "orientation": "h",
        "x": 0,
        "y": 1.04,
        "yanchor": "bottom",
        "font": {"size": 12},
    }
    for chart in charts.values():
        chart.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": INK},
            margin={"t": 44, "l": 12, "r": 14, "b": 10},
            legend=shares_legend,
            separators=", ",
        )
        chart.update_xaxes(automargin=True, gridcolor="#e6edf0", zerolinecolor="#cdd8de")
        chart.update_yaxes(automargin=True, gridcolor="#e6edf0", zerolinecolor="#cdd8de")
    entries = ",".join(f"{json.dumps(key)}:{chart.to_json()}" for key, chart in charts.items())
    payload = ("{" + entries + "}").replace("<", "\\u003c")
    panels = {
        "eda": "".join(
            [
                chart_card("tariffs", "Отток по тарифам", "Доля оттока за шесть месяцев."),
                chart_card("rfm", "Отток по RFM", "Сегменты по пополнениям, приложению и ARPU."),
                chart_card(
                    "traffic",
                    "Как менялся трафик",
                    "Только наблюдения с предыдущим месяцем.",
                    True,
                ),
            ]
        ),
        "model": "".join(
            [
                chart_card("roc", "ROC-кривые", "Ранжирование абонентов по риску оттока."),
                chart_card("pr", "Precision–Recall", "Пунктир показывает долю оттока в тесте."),
                chart_card(
                    "calibration", "Калибровка", "Прогнозируемая и наблюдаемая вероятность."
                ),
                chart_card("importance", "12 факторов оттока", "Permutation importance на тесте."),
            ]
        ),
        "nbo": "".join(
            [
                strategy_table(simulation),
                chart_card("policies", "Маржа и затраты", "Оценка на горизонте трёх месяцев."),
                chart_card("offers", "Назначения по RFM", "Состав офферов в каждом сегменте."),
            ]
        ),
    }
    names = {"eda": "Анализ базы", "model": "Качество модели", "nbo": "Экономика NBO"}
    tabs = "".join(
        f'<button type="button" role="tab" id="tab-{key}" aria-controls="panel-{key}" '
        f'aria-selected="{str(key == "eda").lower()}" tabindex="{0 if key == "eda" else -1}">'
        f"{name}</button>"
        for key, name in names.items()
    )
    content = "".join(
        f'<section class="grid" role="tabpanel" id="panel-{key}" aria-labelledby="tab-{key}"'
        f"{' hidden' if key != 'eda' else ''}>{body}</section>"
        for key, body in panels.items()
    )
    test_month = escape(simulation["test_month"][:7])
    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отток и удержание абонентов</title>
<script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<header class="masthead"><div class="shell">
  <p class="context">Синтетические данные <span aria-hidden="true">/</span>
    Срез {test_month} · отток за {escape(simulation["outcome_month"])}</p>
  <h1>Отток и удержание абонентов</h1>
  <p class="subtitle">Прогноз оттока, персональные офферы и экономика удержания</p>
</div></header>
<main class="shell">
  <dl class="kpis" aria-label="Показатели на тестовом месяце">{cards}</dl>
  <div class="tabs" role="tablist" aria-label="Разделы аналитики">{tabs}</div>
  <noscript><p class="notice">Включите JavaScript, чтобы открыть графики и вкладки.</p></noscript>
  <p class="notice" id="plot-error" role="status" hidden>
    Графики недоступны. Проверьте доступ к CDN Plotly.
  </p>
  {content}
</main>
<footer class="shell"><p>Churn &amp; Next Best Offer · DuckDB / scikit-learn / Plotly</p></footer>
<script type="application/json" id="chart-data">{payload}</script>
<script>{JAVASCRIPT}</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


CSS = """
:root{color-scheme:light;--ink:#1d303b;--muted:#536679;--teal:#087564;--line:#dbe6e9}
*{box-sizing:border-box}
body{margin:0;background:#f1f6f7;color:var(--ink);font:1rem/1.55 'Segoe UI',Arial,sans-serif}
.shell{width:min(1320px,calc(100% - 56px));margin-inline:auto}
.masthead{background:#103732;color:#f2fbf8;padding:42px 0 75px;border-bottom:4px solid #43ba96}
.context{margin:0 0 14px;color:#b4e8d8;font-size:.9rem;letter-spacing:.035em}
.context span{padding-inline:6px;color:#7bcab4}
h1{font-size:clamp(2rem,5vw,3.75rem);letter-spacing:-.045em;line-height:1.04;margin:0;max-width:24ch}
.subtitle{margin:14px 0 0;color:#d1e8e3}
.kpis{margin:-37px 0 29px;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
.kpis div,.card{background:white;border:1px solid var(--line);border-radius:14px}
.kpis div{position:relative;padding:13px 15px;min-width:0;box-shadow:0 4px 16px #14362d0c}
.kpis div:nth-last-child(-n+2){border-top:3px solid #46b698;padding-top:11px}
.kpis dt{font-size:.875rem;color:var(--muted)}
.kpis dd{margin:4px 0 0;font-size:clamp(1.05rem,2.1vw,1.78rem);font-weight:740;
  font-variant-numeric:tabular-nums;letter-spacing:-.035em;white-space:nowrap}
.tabs{display:flex;gap:8px;margin-bottom:18px;border-bottom:1px solid #cadbdf}
.tabs button{font:inherit;font-size:.95rem;font-weight:650;color:var(--muted);background:none;
  border:0;border-bottom:3px solid transparent;padding:10px 15px;cursor:pointer}
.tabs button:hover,.tabs button[aria-selected=true]{color:var(--teal);background:#e8f4f0}
.tabs button[aria-selected=true]{border-bottom-color:var(--teal)}
:focus-visible{outline:2px solid #097a64;outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.card{min-width:0;padding:19px 19px 13px}
.card.wide{grid-column:1/-1}
.card h3{margin:0;font-size:1.125rem;line-height:1.25}
.card p{margin:5px 0 0;color:var(--muted);font-size:.875rem}
.plot{height:320px;min-width:0}
#importance{height:390px}
.table-wrap{overflow:auto;margin-top:14px}
table{width:100%;border-collapse:collapse;font-size:.9rem;white-space:nowrap}
th,td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--line)}
th:first-child{text-align:left}
thead{background:#f1f6f7}
tbody tr:last-child{background:#eaf7f1;color:#095e4a;font-weight:650}
.notice{padding:12px 16px;background:#fff3d9;border-radius:10px;color:#684916}
footer{padding:18px 0;color:var(--muted);font-size:.85rem}
[hidden]{display:none!important}
@media(max-width:1050px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:740px){.shell{width:calc(100% - 28px)}.grid{grid-template-columns:minmax(0,1fr)}
  .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.masthead{padding-top:30px}
  .tabs button{padding:10px 7px;flex:1}.tabs{gap:3px}.card{padding:15px 12px 12px}}
"""

JAVASCRIPT = """
const tabs = Array.from(document.querySelectorAll('[role=tab]'));
const charts = JSON.parse(document.getElementById('chart-data').textContent);
const options = {responsive: true, displaylogo: false, scrollZoom: false};
const drawing = new WeakMap();

async function render(panel) {
  if (panel.hidden) return;
  const notice = document.getElementById('plot-error');
  if (!window.Plotly) {
    notice.hidden = false;
    return;
  }
  try {
    await Promise.all(Array.from(panel.querySelectorAll('.plot'), async plot => {
      if (!drawing.has(plot)) {
        const {data, layout} = charts[plot.id];
        drawing.set(plot, Plotly.newPlot(plot, data, layout, options).catch(error => {
          drawing.delete(plot);
          throw error;
        }));
      }
      await drawing.get(plot);
      if (!panel.hidden) await Plotly.Plots.resize(plot);
    }));
    if (!panel.hidden) notice.hidden = true;
  } catch (error) {
    if (!panel.hidden) notice.hidden = false;
    console.error('Plotly rendering failed', error);
  }
}

function activate(active) {
  for (const tab of tabs) {
    const selected = active === tab;
    tab.setAttribute('aria-selected', selected);
    tab.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    panel.hidden = !selected;
    if (selected) requestAnimationFrame(() => render(panel));
  }
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => activate(tab));
  tab.addEventListener('keydown', event => {
    const positions = {ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1};
    if (!Object.hasOwn(positions, event.key)) return;
    event.preventDefault();
    const next = tabs[(positions[event.key] + tabs.length) % tabs.length];
    next.focus();
    activate(next);
  });
}

document.addEventListener('DOMContentLoaded', () => activate(tabs[0]));
"""
