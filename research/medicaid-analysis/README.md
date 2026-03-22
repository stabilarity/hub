# Medicaid Open Data Analysis — 2018-2024

**Author:** Oleh Ivchenko  
**Data Source:** CMS State Drug Utilization Data (SDUD) via data.medicaid.gov  
**Sample:** 280,000 records (40,000/year × 7 years)  
**Generated:** March 2026  

## What This Is

A data-driven analysis of 7 years of U.S. Medicaid drug spending using official government data. 
Fetches real records directly from the CMS SDUD API and generates 13 publication-quality charts.

## Key Findings

| Metric | Value |
|--------|-------|
| Estimated spending 2018 | $16.1B |
| Estimated spending 2024 | $27.6B |
| Annual growth rate | +$2.1B/year |
| Top 10 drug concentration | 26.6% of all spending |
| Opioid spending share | 1.83% (mostly treatment drugs) |
| Highest fraud susceptibility | AK (59% data suppression) |
| #1 most expensive drug | HUMIRA (CF) — $90.9M in sample |

## Files

```
medicaid-analysis/
├── analysis.py          # Main analysis script — fetches data + generates all charts
├── charts/              # 13 PNG charts (1200x800, 150 DPI, monochrome)
│   ├── 01-annual-spending.png
│   ├── 02-top20-drugs.png
│   ├── 03-state-spending.png
│   ├── 04-price-variance.png
│   ├── 05-opioid-trends.png
│   ├── 06-brand-vs-generic.png
│   ├── 07-state-growth.png
│   ├── 08-volume-vs-cost.png
│   ├── 09-fraud-index.png
│   ├── 10-pareto-concentration.png
│   ├── 11-seasonal-patterns.png
│   ├── 12-price-changes.png
│   └── 13-suppression-rate.png
└── README.md
```

## Running the Analysis

```bash
pip install pandas numpy matplotlib requests
python3 analysis.py
```

Script fetches fresh data from data.medicaid.gov. Runtime ~5 minutes (API throttling).

## API Details

- **API:** CMS Datastore POST endpoint
- **Endpoint:** `https://data.medicaid.gov/api/1/datastore/query/{DATASET_ID}/0`
- **Page size:** 5,000 rows (API maximum)
- **Pages per year:** 8 (= 40,000 rows/year)
- **Total sample:** 280,000 rows

**Dataset IDs by year:**
- 2018: `a1f3598e-fc71-51aa-8560-78e7e1a61b09`
- 2019: `daba7980-e219-5996-9bec-90358fd156f1`
- 2020: `cc318bfb-a9b2-55f3-a924-d47376b32ea3`
- 2021: `eec7fbe6-c4c4-5915-b3d0-be5828ef4e9d`
- 2022: `200c2cba-e58d-4a95-aa60-14b99736808d`
- 2023: `d890d3a9-6b00-43fd-8b31-fcba4c8e2909`
- 2024: `61729e5a-7aa8-448c-8903-ba3e0cd0ea3c`

## Data Schema

Each record contains:
- `utilization_type`: FFSU (Fee-for-Service)
- `state`: 2-letter state code
- `ndc`: National Drug Code
- `product_name`: Drug name
- `year`, `quarter`
- `units_reimbursed`, `number_of_prescriptions`
- `total_amount_reimbursed`, `medicaid_amount_reimbursed`
- `suppression_used`: Whether record is suppressed

## Research Questions Explored

1. **Fraud detection gap** — Is the $175B Medicaid drug system auditable?
2. **Price variance** — Do states pay 10x more for the same drug?
3. **Opioid spending** — Is Medicaid funding addiction or treatment?

## Limitations

- Sample is first 40,000 rows per year (API default order, not random)
- Estimated totals scaled from sample — may have sampling bias
- Opioid classification by keyword — not clinical classification
- Unit prices computed as total/units — varies by formulation
- No fraud data in source — fraud index is proxy metric only

## Design

Charts follow Stabilarity monochrome design system:
- Colors: #000000, #111111, #555555, #bbbbbb, #eeeeee, #ffffff
- Size: 1200×800px equivalent at 150 DPI
- Font: DejaVu Sans