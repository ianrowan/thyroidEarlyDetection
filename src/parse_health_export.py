import argparse
from pathlib import Path
from lxml import etree
import pandas as pd
from tqdm import tqdm
from datetime import datetime

RELEVANT_TYPES = {
    'HKQuantityTypeIdentifierHeartRate': 'heart_rate',
    'HKQuantityTypeIdentifierRestingHeartRate': 'resting_heart_rate',
    'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv_sdnn',
    'HKQuantityTypeIdentifierRespiratoryRate': 'respiratory_rate',
    'HKCategoryTypeIdentifierSleepAnalysis': 'sleep',
    'HKQuantityTypeIdentifierStepCount': 'steps',
    'HKQuantityTypeIdentifierActiveEnergyBurned': 'active_energy',
    'HKQuantityTypeIdentifierOxygenSaturation': 'oxygen_saturation',
    'HKQuantityTypeIdentifierBodyTemperature': 'body_temperature',
}

SLEEP_VALUE_MAP = {
    'HKCategoryValueSleepAnalysisInBed': 'in_bed',
    'HKCategoryValueSleepAnalysisAsleepCore': 'core',
    'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
    'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
    'HKCategoryValueSleepAnalysisAwake': 'awake',
    'HKCategoryValueSleepAnalysisAsleepUnspecified': 'asleep',
    'HKCategoryValueSleepAnalysisAsleep': 'asleep',
}

def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S %z')
    except:
        return None

def parse_export(xml_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {name: [] for name in RELEVANT_TYPES.values()}

    context = etree.iterparse(str(xml_path), events=('end',), tag='Record')

    count = 0
    for event, elem in tqdm(context, desc='Parsing records'):
        record_type = elem.get('type')

        if record_type in RELEVANT_TYPES:
            short_name = RELEVANT_TYPES[record_type]

            start_date = parse_datetime(elem.get('startDate'))
            end_date = parse_datetime(elem.get('endDate'))

            if start_date is None:
                elem.clear()
                continue

            record = {
                'start_date': start_date,
                'end_date': end_date,
                'source': elem.get('sourceName', ''),
                'device': elem.get('device', ''),
            }

            if short_name == 'sleep':
                value = elem.get('value', '')
                record['value'] = SLEEP_VALUE_MAP.get(value, value)
                record['duration_minutes'] = (end_date - start_date).total_seconds() / 60 if end_date else 0
            else:
                try:
                    record['value'] = float(elem.get('value', 0))
                except ValueError:
                    record['value'] = None
                record['unit'] = elem.get('unit', '')

            records[short_name].append(record)

        elem.clear()
        count += 1

        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del context

    print(f"\nProcessed {count:,} total records")
    print("\nRecords per type:")

    for name, data in records.items():
        if data:
            df = pd.DataFrame(data)
            df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'], utc=True)

            output_path = output_dir / f'{name}.parquet'
            df.to_parquet(output_path, index=False)

            date_range = f"{df['start_date'].min().date()} to {df['start_date'].max().date()}"
            print(f"  {name}: {len(df):,} records ({date_range})")

def main():
    parser = argparse.ArgumentParser(description='Parse Apple Health export XML')
    parser.add_argument('--input', type=Path, default=Path('data/apple_health_export/export.xml'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Parsing {args.input} ({args.input.stat().st_size / 1e9:.2f} GB)")
    parse_export(args.input, args.output)
    print(f"\nOutput saved to {args.output}")

if __name__ == '__main__':
    main()
