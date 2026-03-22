import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")
import os, time

CHARTS_DIR = "/root/hub/research/medicaid-analysis/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)
C_BLACK,C_DARK,C_MID,C_LIGHT,C_PALE,C_WHITE = "#000000","#111111","#555555","#bbbbbb","#eeeeee","#ffffff"

PAGE_SIZE = 5000
PAGES = 8

DATASET_IDS = {
    2018:"a1f3598e-fc71-51aa-8560-78e7e1a61b09",
    2019:"daba7980-e219-5996-9bec-90358fd156f1",
    2020:"cc318bfb-a9b2-55f3-a924-d47376b32ea3",
    2021:"eec7fbe6-c4c4-5915-b3d0-be5828ef4e9d",
    2022:"200c2cba-e58d-4a95-aa60-14b99736808d",
    2023:"d890d3a9-6b00-43fd-8b31-fcba4c8e2909",
    2024:"61729e5a-7aa8-448c-8903-ba3e0cd0ea3c",
}

def fetch_dataset(did, pages=PAGES, page_size=PAGE_SIZE):
    all_rows = []
    total_count = 0
    base_url = "https://data.medicaid.gov/api/1/datastore/query/{}/0".format(did)
    for page in range(pages):
        payload = {"limit":page_size,"offset":page*page_size,"results":True,"count":True,"schema":True,"keys":True}
        try:
            r = requests.post(base_url, json=payload, timeout=60)
            if r.status_code == 200:
                d = r.json()
                rows = d.get('results', [])
                if page == 0:
                    total_count = d.get('count', 0)
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < page_size:
                    break
            else:
                print("  HTTP {} on page {}".format(r.status_code, page))
                break
        except Exception as e:
            print("  Error page {}: {}".format(page, e))
            break
        time.sleep(0.2)
    return all_rows, total_count
# Test main
print("Testing paginated fetch...")
rows, count = fetch_dataset("61729e5a-7aa8-448c-8903-ba3e0cd0ea3c", pages=1)
print("rows={}, count={}".format(len(rows), count))
