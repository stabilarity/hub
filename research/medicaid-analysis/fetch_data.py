"""Fetch Medicaid SDUD data from data.medicaid.gov using POST datastore API."""
import requests
import pandas as pd
import numpy as np
import time
import json

DATASET_IDS = {
    2018: 'a1f3598e-fc71-51aa-8560-78e7e1a61b09',
    2019: 'daba7980-e219-5996-9bec-90358fd156f1',
    2020: 'cc318bfb-a9b2-55f3-a924-d47376b32ea3',
    2021: 'eec7fbe6-c4c4-5915-b3d0-be5828ef4e9d',
    2022: '200c2cba-e58d-4a95-aa60-14b99736808d',
    2023: 'd890d3a9-6b00-43fd-8b31-fcba4c8e2909',
    2024: '61729e5a-7aa8-448c-8903-ba3e0cd0ea3c',
}

def fetch_dataset(did, limit=50000):
    url = f'https://data.medicaid.gov/api/1/datastore/query/{did}/0'
    payload = {'limit': limit, 'offset': 0, 'results': True, 'count': True, 'schema': True, 'keys': True}
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            d = r.json()
            return d.get('results', []), d.get('count', 0)
    except Exception as e:
        print(f'  Error: {e}')
    return [], 0

print('Fetching data...')
annual_data = {}
for year, did in DATASET_IDS.items():
    print(f'  {year}...', end=' ', flush=True)
    rows, count = fetch_dataset(did, limit=50000)
    annual_data[year] = {'rows': rows, 'total_rows': count}
    print(f'{len(rows)} rows (total: {count:,})')
    time.sleep(0.5)

# Save summary
with open('/tmp/medicaid_annual_data.json', 'w') as f:
    # Save just metadata
    meta = {str(y): {'total_rows': v['total_rows'], 'sample_rows': len(v['rows'])} for y, v in annual_data.items()}
    json.dump(meta, f)
print('Done. Metadata:', meta)
