import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def load_lab_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    return df

def load_medication_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        date_range = row['Date_Range']
        parts = date_range.split('-')
        if len(parts) >= 2:
            start = parts[0].strip()
            end = parts[1].strip() if parts[1].strip() != 'present' else datetime.now().strftime('%m/%d/%y')
            try:
                start_date = pd.to_datetime(start, format='%m/%d/%y')
                end_date = pd.to_datetime(end, format='%m/%d/%y')
                records.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'medication': row['Med'],
                    'dosage': row['dosage']
                })
            except:
                continue
    return pd.DataFrame(records)

def create_labeling_visualization(features_path: Path, output_path: Path,
                                   lab_path: Path = None, med_path: Path = None):
    df = pd.read_parquet(features_path)
    df['window_start'] = pd.to_datetime(df['window_start'])

    labs = load_lab_results(lab_path) if lab_path and lab_path.exists() else None
    meds = load_medication_history(med_path) if med_path and med_path.exists() else None

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Resting Heart Rate (RHR)',
            'Respiratory Rate',
            'Sleep Duration & Efficiency',
            'HRV (SDNN)'
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    if 'resting_heart_rate_mean' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['window_start'],
                y=df['resting_heart_rate_mean'],
                mode='lines+markers',
                name='RHR Mean',
                line=dict(color='red'),
                hovertemplate='Date: %{x}<br>RHR: %{y:.1f} bpm<extra></extra>'
            ),
            row=1, col=1
        )

        if 'resting_heart_rate_p5' in df.columns and 'resting_heart_rate_p95' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['window_start'].tolist() + df['window_start'].tolist()[::-1],
                    y=df['resting_heart_rate_p5'].tolist() + df['resting_heart_rate_p95'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='RHR Range (5-95%)',
                    showlegend=True
                ),
                row=1, col=1
            )

    if 'respiratory_rate_mean' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['window_start'],
                y=df['respiratory_rate_mean'],
                mode='lines+markers',
                name='Resp Rate Mean',
                line=dict(color='blue'),
                hovertemplate='Date: %{x}<br>Resp Rate: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )

    if 'sleep_total_sleep_minutes' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df['window_start'],
                y=df['sleep_total_sleep_minutes'] / 60,
                name='Sleep Hours',
                marker_color='purple',
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Sleep: %{y:.1f} hrs<extra></extra>'
            ),
            row=3, col=1
        )

    if 'sleep_sleep_efficiency' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['window_start'],
                y=df['sleep_sleep_efficiency'] * 100,
                mode='lines+markers',
                name='Sleep Efficiency %',
                line=dict(color='green'),
                yaxis='y7',
                hovertemplate='Date: %{x}<br>Efficiency: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )

    if 'hrv_sdnn_mean' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['window_start'],
                y=df['hrv_sdnn_mean'],
                mode='lines+markers',
                name='HRV Mean',
                line=dict(color='orange'),
                hovertemplate='Date: %{x}<br>HRV: %{y:.1f} ms<extra></extra>'
            ),
            row=4, col=1
        )

    if labs is not None and len(labs) > 0:
        for _, lab in labs.iterrows():
            for row_num in range(1, 5):
                fig.add_vline(
                    x=lab['Date'],
                    line=dict(color='green', width=2, dash='dash'),
                    row=row_num, col=1
                )

        fig.add_trace(
            go.Scatter(
                x=labs['Date'],
                y=[0] * len(labs),
                mode='markers+text',
                marker=dict(size=10, color='green', symbol='star'),
                text=labs['TSH'].astype(str),
                textposition='top center',
                name='Lab Results (TSH)',
                showlegend=True
            ),
            row=1, col=1
        )

    if meds is not None and len(meds) > 0:
        colors = ['rgba(255,165,0,0.2)', 'rgba(100,149,237,0.2)', 'rgba(144,238,144,0.2)']
        for i, (_, med) in enumerate(meds.iterrows()):
            for row_num in range(1, 5):
                fig.add_vrect(
                    x0=med['start_date'],
                    x1=med['end_date'],
                    fillcolor=colors[i % len(colors)],
                    layer='below',
                    line_width=0,
                    row=row_num, col=1
                )

    fig.update_layout(
        height=1200,
        title_text='Health Metrics for Labeling - Identify Hyperthyroid Episodes',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.update_xaxes(title_text='Date', row=4, col=1)
    fig.update_yaxes(title_text='BPM', row=1, col=1)
    fig.update_yaxes(title_text='Breaths/min', row=2, col=1)
    fig.update_yaxes(title_text='Hours', row=3, col=1)
    fig.update_yaxes(title_text='ms', row=4, col=1)

    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")
    print("Open this file in a browser to interactively explore the data for labeling")

def main():
    parser = argparse.ArgumentParser(description='Create visualization for labeling')
    parser.add_argument('--features', type=Path, default=Path('data/features.parquet'))
    parser.add_argument('--output', type=Path, default=Path('outputs/labeling_viz.html'))
    parser.add_argument('--labs', type=Path, default=Path('data/results.csv'))
    parser.add_argument('--meds', type=Path, default=Path('data/medication_history.csv'))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    create_labeling_visualization(
        args.features,
        args.output,
        args.labs,
        args.meds
    )

if __name__ == '__main__':
    main()
