import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# Rutas relativas a la raíz del proyecto (donde vive este archivo).
BASE_DIR = Path(__file__).resolve().parent

# 1. Cargar el mejor modelo (Random Forest · ROC-AUC: 0.6221) y datos limpios
modelo = joblib.load(BASE_DIR / "modelo_rf.pkl")
df = pd.read_csv(BASE_DIR / "dataset_limpio.csv")

df["tramo_ltv"] = pd.cut(
    df["LTV"],
    bins=[0, 60, 70, 80, 90, 100],
    labels=["≤ 60%", "60% - 70%", "70% - 80%", "80% - 90%", "> 90%"]
)

app = Dash(__name__, title="Fintech Vehicular · Decisión de Crédito (Random Forest)")
server = app.server

CARD_STYLE = {
    "backgroundColor": "#FFFFFF",
    "borderRadius": "10px",
    "padding": "18px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
    "marginBottom": "16px"
}

LABEL_STYLE = {"fontWeight": "600", "fontSize": "13px", "color": "#2C3E50", "marginTop": "10px", "display": "block"}
INPUT_STYLE = {"width": "100%", "padding": "7px", "borderRadius": "6px", "border": "1px solid #CBD5E1", "marginTop": "4px"}


def crear_kpi_card(titulo, valor, subtitulo, color_borde, color_valor="#1E293B"):
    return html.Div(
        style={
            "flex": "1 1 210px",
            "backgroundColor": "#FFFFFF",
            "borderRadius": "10px",
            "padding": "16px 20px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
            "borderTop": f"4px solid {color_borde}"
        },
        children=[
            html.Div(titulo, style={"fontSize": "13px", "fontWeight": "600", "color": "#475569", "marginBottom": "6px"}),
            html.Div(valor, style={"fontSize": "28px", "fontWeight": "800", "color": color_valor, "lineHeight": "1.15", "marginBottom": "6px"}),
            html.Div(subtitulo, style={"fontSize": "12px", "color": "#64748B"})
        ]
    )


app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "backgroundColor": "#F4F6F9", "padding": "20px", "minHeight": "100vh"},
    children=[
        # Encabezado
        html.Div(
            style={**CARD_STYLE, "backgroundColor": "#1F4E79", "color": "white"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.H2("Sistema de Decisión Crediticia · Financiamiento Vehicular", style={"margin": "0 0 6px 0"}),
                            html.P(
                                "Motor predictivo activo: Random Forest (ROC-AUC: 0.6221 · Captura de mora: 63.17%). "
                                "Soporte para Gerencia de Riesgo y Gerencia de Ventas.",
                                style={"margin": 0, "fontSize": "14px", "opacity": "0.92"}
                            )
                        ]),
                    ]
                )
            ]
        ),

        # FILA DE 4 TARJETAS KPI DINÁMICAS (Actualizadas según filtro de cartera)
        html.Div(
            id="contenedor-kpis",
            style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "18px"}
        ),

        # Contenedor principal de 2 columnas
        html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "20px"},
            children=[
                # COLUMNA IZQUIERDA: SIMULADOR DE SOLICITUD
                html.Div(
                    style={"flex": "1 1 380px", "maxWidth": "460px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.H3("1. Datos del Solicitante y del Vehículo", style={"marginTop": 0, "color": "#1F4E79"}),
                                html.P("Complete los datos para determinar si se aprueba o rechaza el préstamo:", style={"fontSize": "13px", "color": "#555"}),

                                html.Label("Tipo de Empleo:", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="inp-emp",
                                    options=[
                                        {"label": "Asalariado (Salaried)", "value": "Salaried"},
                                        {"label": "Independiente (Self employed)", "value": "Self employed"}
                                    ],
                                    value="Salaried",
                                    clearable=False
                                ),

                                html.Label("Monto Solicitado ($):", style=LABEL_STYLE),
                                dcc.Input(id="inp-monto", type="number", value=55000, min=5000, max=300000, step=500, style=INPUT_STYLE),

                                html.Label("Porcentaje Financiado - LTV (%):", style=LABEL_STYLE),
                                dcc.Slider(
                                    id="inp-ltv", min=10, max=98, step=1, value=75,
                                    marks={20: "20%", 50: "50%", 70: "70%", 80: "80%", 90: "90%"},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),

                                html.Label("Puntaje Buró de Crédito (PERFORM_CNS_SCORE):", style=LABEL_STYLE),
                                dcc.Input(id="inp-score", type="number", value=650, min=0, max=900, step=10, style=INPUT_STYLE),

                                html.Label("Categoría Descriptiva Buró (0 = Sin historial, 1-13):", style=LABEL_STYLE),
                                dcc.Input(id="inp-score-desc", type="number", value=10, min=0, max=13, step=1, style=INPUT_STYLE),

                                html.Div(style={"display": "flex", "gap": "10px"}, children=[
                                    html.Div(style={"flex": 1}, children=[
                                        html.Label("Historial Crediticio (meses):", style=LABEL_STYLE),
                                        dcc.Input(id="inp-hist", type="number", value=24, min=0, max=500, step=1, style=INPUT_STYLE),
                                    ]),
                                    html.Div(style={"flex": 1}, children=[
                                        html.Label("Edad Prom. Cuentas (meses):", style=LABEL_STYLE),
                                        dcc.Input(id="inp-age", type="number", value=14, min=0, max=400, step=1, style=INPUT_STYLE),
                                    ])
                                ]),

                                html.Div(style={"display": "flex", "gap": "10px"}, children=[
                                    html.Div(style={"flex": 1}, children=[
                                        html.Label("Cuentas Vencidas Activas:", style=LABEL_STYLE),
                                        dcc.Input(id="inp-overdue", type="number", value=0, min=0, max=30, step=1, style=INPUT_STYLE),
                                    ]),
                                    html.Div(style={"flex": 1}, children=[
                                        html.Label("Moras Últimos 6 Meses:", style=LABEL_STYLE),
                                        dcc.Input(id="inp-delinq", type="number", value=0, min=0, max=20, step=1, style=INPUT_STYLE),
                                    ]),
                                    html.Div(style={"flex": 1}, children=[
                                        html.Label("Consultas Recientes:", style=LABEL_STYLE),
                                        dcc.Input(id="inp-inq", type="number", value=0, min=0, max=40, step=1, style=INPUT_STYLE),
                                    ])
                                ]),

                                html.Button(
                                    "Evaluar Prestamo",
                                    id="btn-evaluar",
                                    n_clicks=0,
                                    style={
                                        "width": "100%", "marginTop": "18px", "padding": "12px",
                                        "backgroundColor": "#1F4E79", "color": "white", "border": "none",
                                        "borderRadius": "6px", "fontWeight": "bold", "cursor": "pointer", "fontSize": "15px"
                                    }
                                )
                            ]
                        )
                    ]
                ),

                # COLUMNA DERECHA: VEREDICTO DE PRÉSTAMO + ANÁLISIS DE CARTERA
                html.Div(
                    style={"flex": "2 1 560px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "10px", "marginBottom": "8px"},
                                    children=[
                                        html.H3("2. Decisión de Otorgamiento (¿Se Presta o No se Presta?)", style={"margin": 0, "color": "#1F4E79"}),
                                        html.Div([
                                            html.Button(
                                                "⬇ Descargar Predicción (CSV)",
                                                id="btn-descargar-csv",
                                                n_clicks=0,
                                                style={
                                                    "backgroundColor": "#0F766E", "color": "white", "border": "none",
                                                    "padding": "8px 14px", "borderRadius": "6px", "fontWeight": "600",
                                                    "fontSize": "13px", "cursor": "pointer"
                                                }
                                            ),
                                            dcc.Download(id="descarga-prediccion-csv")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "20px"},
                                    children=[
                                        html.Div(style={"flex": "1 1 240px"}, children=[dcc.Graph(id="graf-gauge", style={"height": "220px"})]),
                                        html.Div(id="panel-recomendacion", style={"flex": "1 1 280px"})
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap"},
                                    children=[
                                        html.H3("3. Comportamiento Histórico de la Cartera", style={"margin": 0, "color": "#1F4E79"}),
                                        dcc.RadioItems(
                                            id="filtro-empleo",
                                            options=[
                                                {"label": " Todos ", "value": "ALL"},
                                                {"label": " Asalariados ", "value": "Salaried"},
                                                {"label": " Independientes ", "value": "Self employed"}
                                            ],
                                            value="ALL",
                                            inline=True,
                                            style={"fontSize": "13px", "fontWeight": "600"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "12px"},
                                    children=[
                                        html.Div(style={"flex": "1 1 280px"}, children=[dcc.Graph(id="graf-ltv", style={"height": "300px"})]),
                                        html.Div(style={"flex": "1 1 280px"}, children=[dcc.Graph(id="graf-monto", style={"height": "300px"})])
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


def evaluar_cliente(emp, monto, ltv, score, score_desc, hist, age, overdue, delinq, inq):
    monto = float(monto or 55000)
    ltv = float(ltv or 75)
    score = int(score or 0)
    score_desc = int(score_desc or 0)
    hist = int(hist or 0)
    age = int(age or 0)
    overdue = int(overdue or 0)
    delinq = int(delinq or 0)
    inq = int(inq or 0)

    valor_activo = monto / (ltv / 100.0)
    monto_mes_hist = monto / (hist + 1.0)
    tasa_mora_rec = delinq / (overdue + 1.0)

    fila = pd.DataFrame([{
        "LTV": ltv,
        "DISBURSED_AMOUNT": monto,
        "PERFORM_CNS_SCORE": score,
        "PERFORM_CNS_SCORE_DESCRIPTION": score_desc,
        "PRI_OVERDUE_ACCTS": overdue,
        "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": delinq,
        "NO_OF_INQUIRIES": inq,
        "EMPLOYMENT_TYPE": emp,
        "AVERAGE_ACCT_AGE": age,
        "CREDIT_HISTORY_LENGTH": hist,
        "valor_estimado_activo": valor_activo,
        "monto_por_mes_historial": monto_mes_hist,
        "tasa_morosidad_reciente": tasa_mora_rec
    }])

    prob_mora = float(modelo.predict_proba(fila)[0, 1]) * 100
    prob_pago = 100.0 - prob_mora

    if prob_mora < 46:
        veredicto = "SÍ DA PRESTAMO · APROBADO"
        subtitulo = "Riesgo bajo de incumplimiento"
        color = "#2E7D32"
        bg_suave = "#E8F5E9"
        accion_riesgo = f"Probabilidad estimada de pago puntual: {prob_pago:.1f}%. Desembolso autorizado sin restricciones."
        accion_ventas = f"Cerrar venta por ${monto:,.0f} (cuota inicial actual: ${valor_activo - monto:,.0f})."
    elif prob_mora <= 54:
        veredicto = "SÍ PRESTA CON CONDICIONES"
        subtitulo = "Zona gris · Requiere ajustar cuota inicial"
        color = "#C86210"
        bg_suave = "#FFF3E0"
        ltv_objetivo = max(60, int(ltv - 12))
        monto_sugerido = valor_activo * (ltv_objetivo / 100.0)
        accion_riesgo = "No aprobar al LTV actual; riesgo moderado de mora."
        accion_ventas = f"Bajar préstamo a ${monto_sugerido:,.0f} (LTV {ltv_objetivo}%) pidiendo mayor cuota inicial para aprobar."
    else:
        veredicto = "NO DA PRESTAMO · RECHAZADO"
        subtitulo = "Alto riesgo de incumplimiento"
        color = "#B0302A"
        bg_suave = "#FFEBEE"
        accion_riesgo = f"Probabilidad de mora del {prob_mora:.1f}% supera el umbral máximo permitido por Riesgo."
        accion_ventas = "Operación declinada en condiciones actuales. Solicitar codeudor sólido o reducir monto >25%."

    return {
        "fila": fila,
        "valor_activo": valor_activo,
        "ltv": ltv,
        "prob_mora": prob_mora,
        "prob_pago": prob_pago,
        "veredicto": veredicto,
        "subtitulo": subtitulo,
        "color": color,
        "bg_suave": bg_suave,
        "accion_riesgo": accion_riesgo,
        "accion_ventas": accion_ventas
    }


# 3. Callbacks de Interactividad
@app.callback(
    [Output("graf-gauge", "figure"), Output("panel-recomendacion", "children")],
    [Input("btn-evaluar", "n_clicks")],
    [
        State("inp-emp", "value"),
        State("inp-monto", "value"),
        State("inp-ltv", "value"),
        State("inp-score", "value"),
        State("inp-score-desc", "value"),
        State("inp-hist", "value"),
        State("inp-age", "value"),
        State("inp-overdue", "value"),
        State("inp-delinq", "value"),
        State("inp-inq", "value"),
    ]
)
def predecir_riesgo(n_clicks, emp, monto, ltv, score, score_desc, hist, age, overdue, delinq, inq):
    res = evaluar_cliente(emp, monto, ltv, score, score_desc, hist, age, overdue, delinq, inq)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=res["prob_mora"],
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28, "color": res["color"]}},
        title={"text": "Probabilidad de Mora (RF)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": res["color"]},
            "steps": [
                {"range": [0, 46], "color": "#E8F5E9"},
                {"range": [46, 54], "color": "#FFF3E0"},
                {"range": [54, 100], "color": "#FFEBEE"}
            ]
        }
    ))
    fig_gauge.update_layout(margin=dict(t=35, b=10, l=20, r=20), height=215)

    resumen_html = html.Div([
        html.Div(
            style={"backgroundColor": res["color"], "color": "white", "padding": "10px 14px", "borderRadius": "6px", "marginBottom": "10px"},
            children=[
                html.Div(res["veredicto"], style={"fontWeight": "800", "fontSize": "16px", "letterSpacing": "0.3px"}),
                html.Div(res["subtitulo"], style={"fontSize": "12px", "opacity": "0.9"})
            ]
        ),
        html.Div(
            style={"backgroundColor": res["bg_suave"], "padding": "10px", "borderRadius": "6px", "fontSize": "13px"},
            children=[
                html.P([html.Strong("Valor estimado vehículo: "), f"${res['valor_activo']:,.0f} (LTV: {res['ltv']:.0f}%)"], style={"margin": "0 0 6px 0"}),
                html.P([html.Strong("Gerencia de Riesgo: "), res["accion_riesgo"]], style={"margin": "0 0 6px 0"}),
                html.P([html.Strong("Gerencia de Ventas: "), res["accion_ventas"]], style={"margin": 0})
            ]
        )
    ])

    return fig_gauge, resumen_html


@app.callback(
    Output("descarga-prediccion-csv", "data"),
    Input("btn-descargar-csv", "n_clicks"),
    [
        State("inp-emp", "value"),
        State("inp-monto", "value"),
        State("inp-ltv", "value"),
        State("inp-score", "value"),
        State("inp-score-desc", "value"),
        State("inp-hist", "value"),
        State("inp-age", "value"),
        State("inp-overdue", "value"),
        State("inp-delinq", "value"),
        State("inp-inq", "value"),
    ],
    prevent_initial_call=True
)
def descargar_csv_prediccion(n_clicks, emp, monto, ltv, score, score_desc, hist, age, overdue, delinq, inq):
    res = evaluar_cliente(emp, monto, ltv, score, score_desc, hist, age, overdue, delinq, inq)
    df_export = res["fila"].copy()
    df_export["modelo_utilizado"] = "Random Forest (ROC-AUC 0.6221)"
    df_export["probabilidad_mora_pct"] = round(res["prob_mora"], 2)
    df_export["probabilidad_pago_pct"] = round(res["prob_pago"], 2)
    df_export["decision_otorgamiento"] = res["veredicto"]
    df_export["recomendacion_riesgo"] = res["accion_riesgo"]
    df_export["recomendacion_ventas"] = res["accion_ventas"]

    return dcc.send_data_frame(df_export.to_csv, "prediccion_credito_rf.csv", index=False)


@app.callback(
    [Output("contenedor-kpis", "children"), Output("graf-ltv", "figure"), Output("graf-monto", "figure")],
    [Input("filtro-empleo", "value")]
)
def actualizar_cartera(filtro_emp):
    dff = df if filtro_emp == "ALL" else df[df["EMPLOYMENT_TYPE"] == filtro_emp]

    # Cálculo dinámico de los 4 KPIs
    n_creditos = len(dff)
    capital_desembolsado_m = dff["DISBURSED_AMOUNT"].sum() / 1e6
    capital_riesgo_m = dff.loc[dff["LOAN_DEFAULT"] == 1, "DISBURSED_AMOUNT"].sum() / 1e6
    tasa_incumplimiento = dff["LOAN_DEFAULT"].mean() * 100

    kpis_html = [
        crear_kpi_card(
            "Créditos Evaluados (n)",
            f"{n_creditos:,}",
            "Registros filtrados",
            color_borde="#4F46E5",
            color_valor="#0F172A"
        ),
        crear_kpi_card(
            "Capital Desembolsado ($)",
            f"${capital_desembolsado_m:,.1f} M",
            "Volumen total de colocación (Ventas)",
            color_borde="#059669",
            color_valor="#0F172A"
        ),
        crear_kpi_card(
            "Capital en Riesgo ($)",
            f"${capital_riesgo_m:,.1f} M",
            "Monto real en default 1er pago (Riesgo)",
            color_borde="#E11D48",
            color_valor="#E11D48"
        ),
        crear_kpi_card(
            "Tasa de Incumplimiento (%)",
            f"{tasa_incumplimiento:.1f}%",
            "Riesgo histórico de cartera",
            color_borde="#E11D48",
            color_valor="#E11D48"
        )
    ]

    ltv_grp = (
        dff.groupby("tramo_ltv", observed=False)["LOAN_DEFAULT"]
        .mean()
        .mul(100)
        .reset_index()
    )
    fig_ltv = px.bar(
        ltv_grp, x="tramo_ltv", y="LOAN_DEFAULT",
        text=ltv_grp["LOAN_DEFAULT"].round(1).astype(str) + "%",
        title="Tasa de Mora (%) por Tramo de LTV",
        labels={"tramo_ltv": "Tramo LTV", "LOAN_DEFAULT": "Tasa de Mora (%)"},
        color_discrete_sequence=["#1F4E79"]
    )
    fig_ltv.update_traces(textposition="outside")
    fig_ltv.update_yaxes(range=[0, 68])
    fig_ltv.update_layout(template="plotly_white", margin=dict(t=40, b=30, l=40, r=15))

    dff_plot = dff.copy()
    dff_plot["Estado"] = dff_plot["LOAN_DEFAULT"].map({0: "Al día (0)", 1: "En mora (1)"})
    fig_monto = px.box(
        dff_plot, x="Estado", y="DISBURSED_AMOUNT", color="Estado",
        title="Distribución de Monto por Estado",
        labels={"Estado": "", "DISBURSED_AMOUNT": "Monto Desembolsado ($)"},
        color_discrete_map={"Al día (0)": "#1F4E79", "En mora (1)": "#B0302A"}
    )
    fig_monto.update_layout(template="plotly_white", showlegend=False, margin=dict(t=40, b=30, l=40, r=15))

    return kpis_html, fig_ltv, fig_monto


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)