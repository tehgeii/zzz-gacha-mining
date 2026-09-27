"""
Module: visualizer.py
Deskripsi: Komponen visualisasi interaktif berbasis Plotly untuk aplikasi web Streamlit.
Menampilkan kurva probabilitas Rantai Markov, distribusi pity, komparasi algoritma
(Random Forest vs XGBoost), dan analisis kepentingan fitur (Feature Importance).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_PALETTE = {
    "early": "#00FF66",       # Hijau neon
    "mid": "#FFCC00",         # Kuning emas
    "hard": "#FF3366",        # Merah neon / Pink ZZZ
    "background": "#121217",  # Gelap elegan
    "card": "#1E1E26",
    "text": "#E0E0E6",
    "cyan": "#00F0FF",
    "purple": "#A855F7"
}

def plot_markov_cumulative_curve(df_curve, current_pity):
    """
    Kurva probabilitas kumulatif mendapatkan S-Rank dari posisi pity saat ini.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_curve["Tarikan_Tambahan"],
        y=df_curve["Probabilitas_Kumulatif"] * 100,
        mode="lines+markers",
        name="Probabilitas Kumulatif (%)",
        line=dict(color=COLOR_PALETTE["cyan"], width=3),
        marker=dict(size=5, color=COLOR_PALETTE["cyan"]),
        hovertemplate="Tarikan ke-%{x} lagi<br>Total Pity: %{customdata}<br>Peluang: %{y:.1f}%<extra></extra>",
        customdata=df_curve["Total_Pity"]
    ))

    # Garis threshold 50% dan 90%
    fig.add_hline(y=50, line_dash="dash", line_color=COLOR_PALETTE["mid"], annotation_text="Peluang 50%", annotation_position="bottom right")
    fig.add_hline(y=90, line_dash="dash", line_color=COLOR_PALETTE["early"], annotation_text="Peluang 90%", annotation_position="bottom right")

    fig.update_layout(
        title=f"<b>Kurva Probabilitas Kumulatif S-Rank (Markov Chain)</b><br><sup>Mulai dari Pity Saat Ini: {current_pity}</sup>",
        xaxis_title="Jumlah Tarikan Tambahan yang Dibutuhkan",
        yaxis_title="Peluang Kumulatif (%)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_pity_distribution(df_pulls):
    """
    Histogram distribusi pity saat mendapatkan S-Rank.
    """
    s_pulls = df_pulls[df_pulls["rank_type"] == 4]["pity"]
    if len(s_pulls) == 0:
        return None

    fig = go.Figure(data=[go.Histogram(
        x=s_pulls,
        xbins=dict(start=1, end=90, size=5),
        marker_color=COLOR_PALETTE["purple"],
        opacity=0.85
    )])

    fig.update_layout(
        title="<b>Distribusi Tarikan S-Rank Akun (Histogram Pity)</b>",
        xaxis_title="Pity Tarikan",
        yaxis_title="Frekuensi S-Rank",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_model_comparison_bars(clf_metrics, reg_metrics=None):
    """
    Grafik perbandingan akurasi klasifikasi 3 model Data Mining:
    - Absorbing Markov Chain, Random Forest, XGBoost
    """
    models = list(clf_metrics.keys())
    accuracies = [float(str(clf_metrics[m]["Akurasi"]).replace("%", "")) for m in models]
    colors = [COLOR_PALETTE["cyan"], COLOR_PALETTE["purple"], COLOR_PALETTE["early"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        name="Akurasi Klasifikasi (%)",
        marker_color=colors[:len(models)],
        text=[f"{a:.2f}%" for a in accuracies],
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Komparasi Akurasi Klasifikasi (3 Algoritma)</b>",
        yaxis_title="Akurasi (%)",
        yaxis=dict(range=[0, 110]),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_regression_mae_bars(reg_metrics):
    """
    Grafik perbandingan MAE (Mean Absolute Error - Rata-rata Selisih Tarikan):
    Semakin rendah nilainya, semakin presisi prediksinya.
    """
    models = list(reg_metrics.keys())
    maes = [float(reg_metrics[m]["MAE (Rata-rata Selisih Pull)"]) for m in models]
    colors = [COLOR_PALETTE["cyan"], COLOR_PALETTE["purple"], COLOR_PALETTE["early"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=maes,
        name="MAE (Pull Error)",
        marker_color=colors[:len(models)],
        text=[f"{m:.2f} pull" for m in maes],
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Komparasi Kesalahan Regresi (MAE - Selisih Tarikan)</b><br><sup>Semakin rendah nilainya, semakin akurat modelnya</sup>",
        yaxis_title="MAE (Rata-rata Selisih Pull)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_feature_importance(feat_importance_dict):
    """
    Grafik horizontal kepentingan fitur untuk interpretasi model Random Forest.
    """
    rf_feat = feat_importance_dict.get("Random Forest", {})
    if not rf_feat:
        return None

    names_map = {
        "current_pity": "Pity Saat Ini",
        "is_guaranteed": "Status Garansi",
        "prev_s_pity": "Pity S-Rank Sebelumnya",
        "avg_pity_history": "Rata-rata Pity Histori",
        "win_streak": "Streak Kemenangan"
    }

    labels = [names_map.get(k, k) for k in rf_feat.keys()]
    values = list(rf_feat.values())

    df = pd.DataFrame({"Fitur": labels, "Importance": values}).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Fitur"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale="Viridis"
        ),
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Feature Importance (Tingkat Pengaruh Fitur - Random Forest)</b>",
        xaxis_title="Nilai Kepentingan (Importance)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=50, b=40)
    )
    return fig
