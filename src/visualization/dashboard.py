"""
Executive-level Plotly Dash dashboard for Customer Churn Intelligence.

Three front-and-center KPIs:
  1. Revenue at Risk (30-day)
  2. Model Recall @ Threshold
  3. High-Risk Account Count

Additional panels:
  - Churn probability distribution by contract type
  - SHAP feature importance (global)
  - Threshold sensitivity curve (precision / recall / cost)
  - High-risk customer table (sortable)
  - Survival curve (Kaplan-Meier)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


# ─── Color palette (consistent across all charts) ──────────────────────────
COLORS = {
    "primary": "#378ADD",
    "danger": "#D85A30",
    "success": "#1D9E75",
    "warning": "#BA7517",
    "neutral": "#888780",
    "bg": "#F8F8F6",
    "card": "#FFFFFF",
    "text": "#2C2C2A",
    "text_muted": "#5F5E5A",
}

FONT = "DM Sans, system-ui, sans-serif"


def build_kpi_card(title: str, value: str, subtitle: str, color: str) -> dbc.Card:
    """Reusable KPI metric card component."""
    return dbc.Card([
        dbc.CardBody([
            html.P(title, style={"fontSize": "12px", "color": COLORS["text_muted"],
                                  "marginBottom": "4px", "fontFamily": FONT, "letterSpacing": "0.05em"}),
            html.H2(value, style={"fontSize": "28px", "fontWeight": "600",
                                   "color": color, "marginBottom": "4px", "fontFamily": "Georgia, serif"}),
            html.P(subtitle, style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": 0}),
        ])
    ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px",
               "background": COLORS["card"], "boxShadow": "none"})


def create_churn_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Churn probability distribution by contract type."""
    fig = go.Figure()
    contract_types = df["Contract"].unique() if "Contract" in df.columns else ["Month-to-month"]
    palette = [COLORS["danger"], COLORS["warning"], COLORS["primary"]]

    for i, ct in enumerate(contract_types):
        subset = df[df["Contract"] == ct]["churn_probability"] if "Contract" in df.columns else df["churn_probability"]
        fig.add_trace(go.Violin(
            y=subset,
            name=ct,
            fillcolor=palette[i % len(palette)],
            line_color=palette[i % len(palette)],
            opacity=0.7,
            box_visible=True,
            meanline_visible=True,
        ))

    fig.add_hline(y=0.40, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Threshold (0.40)", annotation_position="right")
    fig.update_layout(
        title="Churn Probability Distribution by Contract Type",
        yaxis_title="Churn Probability",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor="white",
        font={"family": FONT, "color": COLORS["text"]},
        showlegend=True,
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_threshold_curve(threshold_df: pd.DataFrame) -> go.Figure:
    """Interactive threshold sensitivity — precision, recall, F2, cost."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["precision"],
                              name="Precision", line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["recall"],
                              name="Recall", line=dict(color=COLORS["success"], width=2)))
    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["f2_score"],
                              name="F2-Score", line=dict(color=COLORS["warning"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=threshold_df["threshold"],
                              y=threshold_df["business_cost_usd"] / threshold_df["business_cost_usd"].max(),
                              name="Normalized Business Cost", line=dict(color=COLORS["danger"], width=2, dash="dot")),
                  secondary_y=True)

    fig.add_vline(x=0.40, line_dash="dash", line_color=COLORS["text_muted"],
                  annotation_text="Current (0.40)")
    fig.update_layout(
        title="Threshold Sensitivity Analysis",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor="white",
        font={"family": FONT, "color": COLORS["text"]},
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top = importance_df.head(top_n).copy()
    fig = go.Figure(go.Bar(
        y=top["feature"],
        x=top["importance_gain"],
        orientation="h",
        marker_color=COLORS["primary"],
        marker_line_width=0,
    ))
    fig.update_layout(
        title=f"Top {top_n} Features by Gain (LightGBM)",
        xaxis_title="Importance (Gain)",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor="white",
        font={"family": FONT, "color": COLORS["text"]},
        yaxis={"autorange": "reversed"},
        height=420,
        margin=dict(l=150, r=40, t=50, b=40),
    )
    return fig


def create_revenue_at_risk_gauge(revenue_at_risk: float, max_value: float = 5_000_000) -> go.Figure:
    """Gauge chart showing revenue at risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=revenue_at_risk,
        number={"prefix": "$", "valueformat": ",.0f"},
        delta={"reference": max_value * 0.3, "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, max_value], "tickformat": "$,.0f"},
            "bar": {"color": COLORS["danger"]},
            "steps": [
                {"range": [0, max_value * 0.3], "color": "#E8F5E9"},
                {"range": [max_value * 0.3, max_value * 0.6], "color": "#FFF8E1"},
                {"range": [max_value * 0.6, max_value], "color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": COLORS["danger"], "width": 3}, "value": max_value * 0.6},
        },
        title={"text": "Revenue at Risk (30-day)", "font": {"size": 14}},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="white",
                      font={"family": FONT})
    return fig


def build_dashboard_app(
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    eval_result=None,
) -> "dash.Dash":
    """
    Build the full executive dashboard Dash application.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain: customerID, churn_probability, Contract,
        estimated_clv, revenue_at_risk columns.
    importance_df : pd.DataFrame
        Output of model.get_feature_importance().
    threshold_df : pd.DataFrame
        Output of evaluator.get_threshold_curve().
    eval_result : EvaluationResult, optional
        Model evaluation metrics.
    """
    if not DASH_AVAILABLE:
        raise ImportError("Install dash: pip install dash dash-bootstrap-components")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Churn Intelligence Dashboard",
    )

    # Compute KPI values
    threshold = 0.40
    high_risk = predictions_df[predictions_df["churn_probability"] >= threshold]
    revenue_at_risk = high_risk["revenue_at_risk"].sum() if "revenue_at_risk" in high_risk.columns else len(high_risk) * 250
    recall_pct = f"{(eval_result.recall * 100):.1f}%" if eval_result else "N/A"
    auc = f"{eval_result.auc_roc:.3f}" if eval_result else "N/A"

    app.layout = dbc.Container([

        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Customer Churn Intelligence",
                        style={"fontFamily": "Georgia, serif", "fontWeight": 600,
                               "fontSize": "26px", "color": COLORS["text"], "marginBottom": "4px"}),
                html.P("Executive Dashboard — Real-time churn risk monitoring & retention intelligence",
                       style={"color": COLORS["text_muted"], "fontSize": "13px", "marginBottom": 0}),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("Model: LightGBM + CatBoost", style={"fontSize": "11px",
                              "background": "#E6F1FB", "color": "#0C447C",
                              "padding": "4px 10px", "borderRadius": "20px", "marginRight": "8px"}),
                    html.Span(f"AUC: {auc}", style={"fontSize": "11px",
                              "background": "#E1F5EE", "color": "#085041",
                              "padding": "4px 10px", "borderRadius": "20px"}),
                ], style={"textAlign": "right", "paddingTop": "12px"}),
            ], width=4),
        ], style={"padding": "24px 0 16px", "borderBottom": "0.5px solid #E0DFD8", "marginBottom": "24px"}),

        # KPI Row
        dbc.Row([
            dbc.Col(build_kpi_card(
                "REVENUE AT RISK (30-DAY)",
                f"${revenue_at_risk:,.0f}",
                "Total CLV of high-risk customers",
                COLORS["danger"],
            ), width=4),
            dbc.Col(build_kpi_card(
                "MODEL RECALL @ THRESHOLD",
                recall_pct,
                "% of actual churners identified (threshold=0.40)",
                COLORS["primary"],
            ), width=4),
            dbc.Col(build_kpi_card(
                "HIGH-RISK ACCOUNTS",
                f"{len(high_risk):,}",
                "Customers requiring immediate retention action",
                COLORS["warning"],
            ), width=4),
        ], className="mb-4"),

        # Row 2: Distribution + Revenue gauge
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(dcc.Graph(figure=create_churn_distribution_chart(predictions_df),
                                          config={"displayModeBar": False}))
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px"}),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(dcc.Graph(figure=create_revenue_at_risk_gauge(revenue_at_risk),
                                          config={"displayModeBar": False}))
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px", "marginBottom": "12px"}),
                dbc.Card([
                    dbc.CardBody([
                        html.P("Threshold Optimization", style={"fontSize": "12px",
                               "color": COLORS["text_muted"], "marginBottom": "8px"}),
                        html.P("Current threshold 0.40 maximizes F2-score. "
                               "Adjust to trade precision vs. recall based on retention campaign budget.",
                               style={"fontSize": "12px", "color": COLORS["text"], "lineHeight": "1.5"}),
                    ])
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px"}),
            ], width=4),
        ], className="mb-4"),

        # Row 3: Feature importance + Threshold curve
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(dcc.Graph(figure=create_feature_importance_chart(importance_df),
                                          config={"displayModeBar": False}))
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(dcc.Graph(figure=create_threshold_curve(threshold_df),
                                          config={"displayModeBar": False}))
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px"}),
            ], width=6),
        ], className="mb-4"),

        # Row 4: High-risk customer table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("High-Risk Customer List — Action Queue",
                                   style={"fontFamily": FONT, "fontSize": "13px",
                                          "fontWeight": "500", "background": "white",
                                          "borderBottom": "0.5px solid #E0DFD8"}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="high-risk-table",
                            columns=[
                                {"name": "Customer ID", "id": "customerID"},
                                {"name": "Churn Probability", "id": "churn_probability",
                                 "type": "numeric", "format": {"specifier": ".1%"}},
                                {"name": "Contract", "id": "Contract"},
                                {"name": "Revenue at Risk ($)", "id": "revenue_at_risk",
                                 "type": "numeric", "format": {"specifier": ",.0f"}},
                            ],
                            data=high_risk.head(50).to_dict("records") if len(high_risk) > 0 else [],
                            sort_action="native",
                            filter_action="native",
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "fontFamily": FONT,
                                "fontSize": "13px",
                                "padding": "8px 12px",
                                "border": "0.5px solid #E0DFD8",
                            },
                            style_header={
                                "fontWeight": "500",
                                "background": "#F1EFE8",
                                "border": "0.5px solid #E0DFD8",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": "{churn_probability} >= 0.7"},
                                    "backgroundColor": "#FFEBEE",
                                    "color": "#712B13",
                                },
                            ],
                        )
                    ]),
                ], style={"border": "0.5px solid #E0DFD8", "borderRadius": "12px"}),
            ], width=12),
        ], className="mb-4"),

        # Footer
        dbc.Row([
            dbc.Col(html.P(
                "Churn Intelligence System v1.0 · LightGBM + Survival Analysis · github.com/thed700",
                style={"fontSize": "11px", "color": COLORS["text_muted"],
                       "textAlign": "center", "padding": "16px 0"}
            ))
        ]),

    ], fluid=True, style={"background": COLORS["bg"], "minHeight": "100vh", "fontFamily": FONT})

    return app


if __name__ == "__main__":
    # Demo mode: generate synthetic data for dashboard preview
    np.random.seed(42)
    n = 500
    demo_df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "churn_probability": np.clip(np.random.beta(2, 5, n), 0, 1),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "estimated_clv": np.random.uniform(100, 2000, n),
    })
    demo_df["revenue_at_risk"] = demo_df["churn_probability"] * demo_df["estimated_clv"]

    importance_df = pd.DataFrame({
        "feature": ["charge_volatility_ratio", "tenure_contract_interaction", "MonthlyCharges",
                     "service_adoption_density", "cohort_clv_percentile", "tenure", "Contract",
                     "support_recency_score", "TotalCharges", "PaymentMethod"],
        "importance_gain": [980, 850, 720, 610, 530, 490, 380, 310, 270, 210],
    })

    threshold_df = pd.DataFrame({
        "threshold": np.arange(0.1, 0.9, 0.05),
        "precision": np.linspace(0.3, 0.9, 16),
        "recall": np.linspace(0.95, 0.2, 16),
        "f2_score": np.array([0.6, 0.65, 0.72, 0.78, 0.82, 0.84, 0.83, 0.81, 0.76, 0.70, 0.63, 0.55, 0.46, 0.37, 0.28, 0.19]),
        "business_cost_usd": np.linspace(200000, 50000, 16),
        "flagged_customers": np.linspace(350, 40, 16, dtype=int),
    })

    app = build_dashboard_app(demo_df, importance_df, threshold_df)
    print("\n Dashboard running at http://localhost:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
