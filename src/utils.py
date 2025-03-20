import pandas as pd
from functools import lru_cache
import re

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

def split_into_chunks(lst, chunk_size=72):
    return [lst[i:i + chunk_size] for i in range(len(lst) - chunk_size + 1)]

def create_test_dataset(df, len_chunks):
    # test_df = pd.DataFrame(columns=['target', 'start', 'feat_static_cat', 'feat_dynamic_real'])
    to_add = []
    for row in df.itertuples():
        tmp = pd.DataFrame()
        tmp['target'] = split_into_chunks(row.target, len_chunks)
        tmp['start'] = [row.start + datetime.timedelta(hours=i) for i in range(len(tmp))]
        tmp['feat_static_cat'] = [row.feat_static_cat] * len(tmp)
        tmp['feat_dynamic_real'] = row.feat_dynamic_real

        to_add.append(tmp)
    final_df = pd.concat(to_add, axis=0).reset_index(drop=True)
    final_df['item_id'] = ['T'+str(i+1) for i in range(len(final_df))]
    return final_df

def scale_series(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)

def create_sequences(data, input_sequence_length, output_sequence_length):
    X, y = [], []
    for i in range(len(data) - input_sequence_length - output_sequence_length + 1):
        X.append(data[i:(i + input_sequence_length)])
        y.append(data[(i + input_sequence_length):(i + input_sequence_length + output_sequence_length)])
    return np.array(X), np.array(y)


def parse_frequency(freq):
    m = re.match(r'(\d+)?([A-Za-z]+)', freq)
    if not m:
        raise ValueError("Invalid frequency format")

    num = int(m.group(1)) if m.group(1) else 1
    unit = m.group(2).upper()

    if unit in ['Y', 'YEAR', 'YEARS']:
        return {'days': num * 365}
    if unit in ['M', 'MO', 'MONTH', 'MONTHS']:
        return {'days': num * 30}

    # Mapping for supported units to timedelta keyword arguments.
    unit_mapping = {
        'W': 'weeks',
        'D': 'days',
        'H': 'hours',
        'T': 'minutes',
        'S': 'seconds', 
        'MS': 'milliseconds',
        'US': 'microseconds'
    }

    if unit not in unit_mapping:
        raise ValueError(f"Unsupported time unit: {unit}")

    return {unit_mapping[unit]: num}
