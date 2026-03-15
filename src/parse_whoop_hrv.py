import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_whoop_hrv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    hrv_df = df[['Cycle start time', 'Heart rate variability (ms)']].copy()
    hrv_df.columns = ['start_date', 'value']
    hrv_df = hrv_df.dropna(subset=['value'])

    hrv_df['start_date'] = pd.to_datetime(hrv_df['start_date'], utc=True)
    hrv_df['end_date'] = hrv_df['start_date'] + pd.Timedelta(minutes=5)
    hrv_df['source'] = 'Whoop'
    hrv_df['device'] = 'Whoop 4.0'
    hrv_df['unit'] = 'ms'
    hrv_df['hrv_type'] = 'rmssd'

    return hrv_df[['start_date', 'end_date', 'source', 'device', 'value', 'unit', 'hrv_type']]

def create_unified_hrv(sdnn_path: Path, whoop_csv_path: Path, output_path: Path):
    sdnn_df = pd.read_parquet(sdnn_path)
    sdnn_df['start_date'] = pd.to_datetime(sdnn_df['start_date'], utc=True)
    sdnn_df['hrv_type'] = 'sdnn'

    rmssd_df = parse_whoop_hrv(whoop_csv_path)

    print(f"SDNN records: {len(sdnn_df)} ({sdnn_df['start_date'].min().date()} to {sdnn_df['start_date'].max().date()})")
    print(f"RMSSD records: {len(rmssd_df)} ({rmssd_df['start_date'].min().date()} to {rmssd_df['start_date'].max().date()})")

    sdnn_mean = sdnn_df['value'].mean()
    sdnn_std = sdnn_df['value'].std()
    rmssd_mean = rmssd_df['value'].mean()
    rmssd_std = rmssd_df['value'].std()

    print(f"\nNormalization stats:")
    print(f"  SDNN:  mean={sdnn_mean:.1f}ms, std={sdnn_std:.1f}ms")
    print(f"  RMSSD: mean={rmssd_mean:.1f}ms, std={rmssd_std:.1f}ms")

    sdnn_df['value_raw'] = sdnn_df['value']
    sdnn_df['value'] = (sdnn_df['value'] - sdnn_mean) / sdnn_std

    rmssd_df['value_raw'] = rmssd_df['value']
    rmssd_df['value'] = (rmssd_df['value'] - rmssd_mean) / rmssd_std

    unified_df = pd.concat([sdnn_df, rmssd_df], ignore_index=True)
    unified_df = unified_df.sort_values('start_date').reset_index(drop=True)

    print(f"\nUnified HRV records: {len(unified_df)}")
    print(f"Date range: {unified_df['start_date'].min().date()} to {unified_df['start_date'].max().date()}")
    print(f"Z-scored value stats: mean={unified_df['value'].mean():.3f}, std={unified_df['value'].std():.3f}")

    unified_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return unified_df

def main():
    parser = argparse.ArgumentParser(description='Parse Whoop HRV and create unified HRV dataset')
    parser.add_argument('--whoop-csv', type=Path, default=Path('data/physiological_cycles.csv'))
    parser.add_argument('--sdnn-parquet', type=Path, default=Path('data/processed/hrv_sdnn.parquet'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/hrv_unified.parquet'))
    args = parser.parse_args()

    create_unified_hrv(args.sdnn_parquet, args.whoop_csv, args.output)

if __name__ == '__main__':
    main()
