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

VALID_STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC']

OPIOID_KW = ['OXYCODONE','HYDROCODONE','MORPHINE','FENTANYL','CODEINE','TRAMADOL',
    'METHADONE','BUPRENORPHINE','NALOXONE','SUBOXONE','OXYCONTIN','VICODIN',
    'PERCOCET','NORCO','DILAUDID','HYDROMORPHONE','SUBUTEX','ZUBSOLV','BUTRANS',
    'EMBEDA','KADIAN','NUCYNTA','TAPENTADOL','BELBUCA']

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
                print("  HTTP {} page {}".format(r.status_code, page))
                break
        except Exception as e:
            print("  Error page {}: {}".format(page, e))
            break
        time.sleep(0.15)
    return all_rows, total_count

def is_opioid(name):
    n = str(name).upper()
    return any(k in n for k in OPIOID_KW)

def chart_style():
    plt.rcParams.update({
        "figure.facecolor":C_WHITE,"axes.facecolor":C_WHITE,"axes.edgecolor":C_DARK,
        "axes.labelcolor":C_DARK,"xtick.color":C_MID,"ytick.color":C_MID,
        "text.color":C_DARK,"grid.color":C_PALE,"grid.linewidth":0.5,
        "font.family":"DejaVu Sans","font.size":11,
        "axes.titlesize":12,"axes.titleweight":"bold","axes.titlepad":10,
    })

def save_chart(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    print("  Saved: "+filename)

print("="*55)
print("MEDICAID OPEN DATA ANALYSIS 2018-2024")
print("Source: data.medicaid.gov SDUD API")
print("="*55)
print()
print("[DATA] Fetching 8 pages x 5000 rows = 40k rows/year...")
annual_data = {}
for year,did in DATASET_IDS.items():
    print("  {}... ".format(year), end="", flush=True)
    rows,count = fetch_dataset(did)
    annual_data[year] = {"rows":rows,"total_rows":count}
    print("{} rows (DB: {:,})".format(len(rows), count))

dfs = {}
for year,info in annual_data.items():
    if info['rows']:
        df = pd.DataFrame(info['rows'])
        for col in ['total_amount_reimbursed','medicaid_amount_reimbursed','units_reimbursed','number_of_prescriptions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['year'] = year
        dfs[year] = df

if not dfs:
    print("ERROR: No data. Exiting.")
    exit(1)

combined = pd.concat(dfs.values(), ignore_index=True)
combined['unit_price'] = np.where(combined['units_reimbursed']>0,
    combined['total_amount_reimbursed']/combined['units_reimbursed'], np.nan)
combined['is_opioid'] = combined['product_name'].apply(is_opioid)
print("Combined: {:,} rows across {} years".format(len(combined), len(dfs)))

chart_style()
years_list = sorted(dfs.keys())

drug_totals = combined.groupby('product_name').agg(
    total_spending=('total_amount_reimbursed','sum'),
    total_rxs=('number_of_prescriptions','sum'),
    total_units=('units_reimbursed','sum')
).reset_index()
drug_totals['product_name'] = drug_totals['product_name'].str.strip()
drug_totals = drug_totals[drug_totals['product_name']!=''].sort_values('total_spending',ascending=False)

est_billions = []
for y in years_list:
    df = dfs[y]
    sample_total = df['total_amount_reimbursed'].sum()
    total_rows = annual_data[y]['total_rows']
    scale = total_rows/len(df) if len(df)>0 else 1
    est_billions.append(sample_total*scale/1e9)
print()
print("Estimated annual spending (scaled):")
for y,b in zip(years_list,est_billions):
    print("  {}: ${:.2f}B".format(y,b))
z_coef = np.polyfit(years_list, est_billions, 1)
z_poly = np.poly1d(z_coef)
print()
print("[01] Annual Spending...")
fig,ax = plt.subplots(figsize=(12,8))
clr01 = [C_MID if y<2024 else C_BLACK for y in years_list]
bars01 = ax.bar(years_list, est_billions, color=clr01, width=0.6, edgecolor=C_DARK, linewidth=0.8)
for bar,val in zip(bars01,est_billions):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(est_billions)*0.012,
        '${:.1f}B'.format(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
xline = np.linspace(min(years_list),max(years_list),100)
ax.plot(xline, z_poly(xline), '--', color=C_LIGHT, linewidth=1.5, label='Trend (+${:.1f}B/yr)'.format(z_coef[0]))
ax.legend(fontsize=10)
ax.set_title("America's Medicaid Drug Bill: 7 Years of Relentless Growth\n(Estimated Total Reimbursements 2018-2024, Scaled from 40k-Row API Sample)")
ax.set_xlabel('Year'); ax.set_ylabel('Estimated Spending ($ Billions)')
ax.set_ylim(0, max(est_billions)*1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: '${:.0f}B'.format(x)))
ax.grid(axis='y', alpha=0.4); ax.set_xticks(years_list)
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov | Sample scaled to full dataset size',
    transform=ax.transAxes, fontsize=8, color=C_LIGHT, va='bottom')
plt.tight_layout(); save_chart(fig,'01-annual-spending.png')
print("[02] Top 20 Drugs...")
top20 = drug_totals.head(20)
fig,ax = plt.subplots(figsize=(12,10))
clr02 = [C_BLACK if i<5 else C_MID if i<10 else C_LIGHT for i in range(len(top20))]
ax.barh(range(len(top20)), top20['total_spending'].values/1e6, color=clr02, edgecolor=C_DARK, linewidth=0.5)
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20['product_name'].values, fontsize=9)
ax.invert_yaxis()
ax.set_title('Top 20 Medicaid Drug Money Sinks: Where Your Tax Dollars Go\n(Cumulative Sample Reimbursements 2018-2024)')
ax.set_xlabel('Total Reimbursed ($ Millions, Sample)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: '${:,.0f}M'.format(x)))
ax.grid(axis='x', alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'02-top20-drugs.png')
print("[03] State Spending...")
state_tot = combined[combined['state'].isin(VALID_STATES)].groupby('state').agg(
    total_spending=('total_amount_reimbursed','sum')).reset_index().sort_values('total_spending',ascending=True).tail(35)
fig,ax = plt.subplots(figsize=(12,11))
n03=len(state_tot)
clr03=[C_BLACK if i>=n03-5 else C_MID if i>=n03-15 else C_LIGHT for i in range(n03)]
ax.barh(range(n03), state_tot['total_spending'].values/1e6, color=clr03, edgecolor=C_DARK, linewidth=0.4)
ax.set_yticks(range(n03)); ax.set_yticklabels(state_tot['state'].values, fontsize=8)
ax.set_title('The Medicaid Spending Map: Drug Reimbursements by State\n(Sample Totals 2018-2024)')
ax.set_xlabel('Total Drug Spending ($ Millions, Sample)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: '${:,.0f}M'.format(x)))
ax.grid(axis='x', alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'03-state-spending.png')
print("[04] Price Variance...")
top5d = drug_totals.sort_values('total_rxs',ascending=False).head(15)['product_name'].values[:5].tolist()
print("  Top 5 by Rx: {}".format(top5d))
pd4 = combined[combined['product_name'].isin(top5d)&combined['unit_price'].notna()&(combined['unit_price']>0)].copy()
cl4=[]
for drug in top5d:
    dd=pd4[pd4['product_name']==drug]
    if len(dd)>0:
        q99=dd['unit_price'].quantile(0.99)
        cl4.append(dd[dd['unit_price']<=q99])
pd4 = pd.concat(cl4) if cl4 else pd4
fig,axes04=plt.subplots(1,len(top5d),figsize=(14,7))
if not hasattr(axes04,'__len__'): axes04=[axes04]
for i,(drug,ax) in enumerate(zip(top5d,axes04)):
    dp=pd4[pd4['product_name']==drug]['unit_price'].dropna()
    if len(dp)<5:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes); continue
    mn04,mx04=dp.min(),dp.max()
    ratio04=mx04/mn04 if mn04>0 else 0
    ax.boxplot(dp.values, patch_artist=True,
        boxprops=dict(facecolor=C_PALE,color=C_DARK),
        medianprops=dict(color=C_BLACK,linewidth=2),
        whiskerprops=dict(color=C_MID),capprops=dict(color=C_MID),
        flierprops=dict(marker='.',color=C_LIGHT,alpha=0.3,markersize=3))
    ax.set_title(drug.strip()[:14],fontsize=8,fontweight='bold')
    ax.set_ylabel('Unit Price ($)' if i==0 else '',fontsize=9)
    ax.tick_params(labelbottom=False)
    ax.text(0.5,0.02,'Max/Min\n{:.0f}x'.format(ratio04),transform=ax.transAxes,ha='center',fontsize=8,color=C_MID)
    ax.grid(axis='y',alpha=0.3)
fig.suptitle('The Price Variance Scandal: Same Drug, Different State, Wildly Different Medicaid Cost\n(Unit Price Distribution 2018-2024 Sample)',fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout(); save_chart(fig,'04-price-variance.png')
print("[05] Opioid Trends...")
ot = combined.groupby(['year','is_opioid'])['total_amount_reimbursed'].sum().reset_index()
op = ot.pivot(index='year',columns='is_opioid',values='total_amount_reimbursed').fillna(0)
op.columns=['non_opioid','opioid']
op['pct']=op['opioid']/(op['opioid']+op['non_opioid'])*100
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,9),sharex=True)
ax1.plot(op.index,op['non_opioid']/1e6,'o-',color=C_DARK,linewidth=2,markersize=6,label='Non-Opioid')
ax1.plot(op.index,op['opioid']/1e6,'s--',color=C_MID,linewidth=2,markersize=6,label='Opioid-Related')
ax1.set_ylabel('Spending ($ Millions, Sample)'); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: '${:,.0f}M'.format(x)))
ax1.set_title("Medicaid's Opioid Dependency: Tracking the Crisis in Spending\n(Opioid vs Non-Opioid Reimbursements, 2018-2024 Sample)")
ax2.bar(op.index,op['pct'],color=C_MID,width=0.6,edgecolor=C_DARK)
ax2.set_ylabel('Opioid % of Total'); ax2.set_xlabel('Year')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: '{:.2f}%'.format(x)))
ax2.grid(axis='y',alpha=0.3)
for x,pct in zip(op.index,op['pct']):
    ax2.text(x,pct+0.04,'{:.2f}%'.format(pct),ha='center',fontsize=9)
ax2.text(0.01,0.01,'Source: CMS SDUD | Opioids by drug name keywords',transform=ax2.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'05-opioid-trends.png')
print("[06] Brand vs Generic...")
dc = combined[combined['unit_price'].notna()&(combined['unit_price']>0)].groupby('product_name').agg(
    avg_price=('unit_price','mean'),total_spend=('total_amount_reimbursed','sum'),total_rxs=('number_of_prescriptions','sum')).reset_index()
dc=dc[dc['total_rxs']>100]
q25p=dc['avg_price'].quantile(0.25); q75p=dc['avg_price'].quantile(0.75)
dc['cat']='Mid-Range'
dc.loc[dc['avg_price']<=q25p,'cat']='Low-Cost (Generic)'
dc.loc[dc['avg_price']>=q75p,'cat']='High-Cost (Brand)'
cs=dc.groupby('cat').agg(avg_price=('avg_price','mean'),total_spend=('total_spend','sum')).reset_index()
cats06=['Low-Cost (Generic)','Mid-Range','High-Cost (Brand)']
cp=[cs[cs['cat']==c]['avg_price'].values[0] if c in cs['cat'].values else 0 for c in cats06]
csp=[cs[cs['cat']==c]['total_spend'].values[0]/1e6 if c in cs['cat'].values else 0 for c in cats06]
r06=cp[2]/cp[0] if cp[0]>0 else 0
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
b1=ax1.bar(cats06,cp,color=[C_PALE,C_MID,C_BLACK],edgecolor=C_DARK,width=0.5)
for bar,val in zip(b1,cp):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(cp)*0.02,'${:.2f}'.format(val),ha='center',fontsize=11,fontweight='bold')
ax1.set_title('Avg Unit Price by Category',fontsize=12,fontweight='bold'); ax1.set_ylabel('Avg Price/Unit ($)'); ax1.grid(axis='y',alpha=0.3)
b2=ax2.bar(cats06,csp,color=[C_PALE,C_MID,C_BLACK],edgecolor=C_DARK,width=0.5)
for bar,val in zip(b2,csp):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(csp)*0.02,'${:,.0f}M'.format(val),ha='center',fontsize=10,fontweight='bold')
ax2.set_title('Total Spending by Category',fontsize=12,fontweight='bold')
ax2.set_ylabel('Spending ($ Millions, Sample)'); ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:'${:,.0f}M'.format(x))); ax2.grid(axis='y',alpha=0.3)
fig.suptitle('Brand vs Generic: High-Cost Drugs Are {:.0f}x More Expensive Per Unit\n(Price Quartile Analysis, SDUD Sample 2018-2024)'.format(r06),fontsize=12,fontweight='bold')
ax1.text(0.01,0.01,'Source: CMS SDUD | Categories by price quartile',transform=ax1.transAxes,fontsize=7,color=C_LIGHT)
plt.tight_layout(); save_chart(fig,'06-brand-vs-generic.png')
print("[07] State Growth Rate...")
sy=combined[combined['state'].isin(VALID_STATES)].groupby(['state','year'])['total_amount_reimbursed'].sum().reset_index()
sp=sy.pivot(index='state',columns='year',values='total_amount_reimbursed').fillna(0)
yr1c,yr2c=min(sp.columns),max(sp.columns)
sp['growth']=((sp[yr2c]-sp[yr1c])/sp[yr1c].replace(0,np.nan))*100
sg=sp[['growth']].dropna().sort_values('growth',ascending=True)
fig,ax=plt.subplots(figsize=(12,10))
n07=len(sg)
clr07=[C_BLACK if v>100 else C_DARK if v>50 else C_MID if v>0 else C_LIGHT for v in sg['growth'].values]
ax.barh(range(n07),sg['growth'].values,color=clr07,edgecolor=C_DARK,linewidth=0.4)
ax.set_yticks(range(n07)); ax.set_yticklabels(sg.index.values,fontsize=8)
ax.axvline(x=0,color=C_DARK,linewidth=1)
for i,val in enumerate(sg['growth'].values):
    ax.text(val+1 if val>=0 else val-1,i,'{:.0f}%'.format(val),va='center',fontsize=7,ha='left' if val>=0 else 'right')
ax.set_xlabel('Spending Growth (%)')
ax.set_title('The Fastest-Growing Medicaid Drug Markets: Who Is Accelerating?\n(Drug Spending Growth {}>{} by State, SDUD Sample)'.format(yr1c,yr2c))
ax.grid(axis='x',alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'07-state-growth.png')
print("[08] Volume vs Cost...")
ds08=drug_totals.copy()
ds08=ds08[ds08['total_rxs']>500].copy()
ds08['avg_price']=np.where(ds08['total_units']>0,ds08['total_spending']/ds08['total_units'],np.nan)
p99=ds08['avg_price'].quantile(0.99)
ds08=ds08[ds08['avg_price']<=p99].nlargest(100,'total_spending')
fig,ax=plt.subplots(figsize=(12,8))
sz08=np.clip(ds08['avg_price'].fillna(1),1,1000)*2
sc08=ax.scatter(ds08['total_rxs']/1e3,ds08['total_spending']/1e6,s=sz08,
    c=np.log1p(ds08['avg_price'].fillna(0)),cmap='Greys',alpha=0.7,edgecolors=C_DARK,linewidths=0.5)
cb08=plt.colorbar(sc08,ax=ax); cb08.set_label('Log(Avg Unit Price)',fontsize=9)
for _,row in ds08.nlargest(8,'total_spending').iterrows():
    ax.annotate(row['product_name'][:14],(row['total_rxs']/1e3,row['total_spending']/1e6),
        fontsize=7,color=C_DARK,xytext=(5,5),textcoords='offset points')
ax.set_xlabel('Total Prescriptions (Thousands, Sample)'); ax.set_ylabel('Total Spending ($ Millions, Sample)')
ax.set_title('Volume vs Cost: The Drugs Milking Medicaid\n(Bubble size = Avg Unit Price; top 100 drugs 2018-2024 sample)')
ax.grid(alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'08-volume-vs-cost.png')
print("[09] Fraud Susceptibility Index...")
sf09=combined[combined['state'].isin(VALID_STATES)].copy()
sf09['suppressed']=(sf09['suppression_used'].astype(str).str.lower()=='true').astype(int)
fm=sf09.groupby('state').agg(total_spend=('total_amount_reimbursed','sum'),
    avg_price=('unit_price','mean'),supp_rate=('suppressed','mean'),price_std=('unit_price','std')).reset_index().dropna()
for col09 in ['total_spend','avg_price','supp_rate','price_std']:
    mn09,mx09=fm[col09].min(),fm[col09].max()
    fm[col09+'_n']=(fm[col09]-mn09)/(mx09-mn09) if mx09>mn09 else 0.0
fm['fraud_idx']=(fm['total_spend_n']*0.3+fm['avg_price_n']*0.25+fm['supp_rate_n']*0.25+fm['price_std_n']*0.2)*100
fm=fm.sort_values('fraud_idx',ascending=True)
fig,ax=plt.subplots(figsize=(12,10))
n09=len(fm)
clr09=[C_BLACK if v>70 else C_DARK if v>50 else C_MID if v>30 else C_LIGHT for v in fm['fraud_idx'].values]
ax.barh(range(n09),fm['fraud_idx'].values,color=clr09,edgecolor=C_DARK,linewidth=0.4)
ax.set_yticks(range(n09)); ax.set_yticklabels(fm['state'].values,fontsize=8)
for i,val in enumerate(fm['fraud_idx'].values):
    ax.text(val+0.5,i,'{:.1f}'.format(val),va='center',fontsize=7)
ax.axvline(x=50,color=C_MID,linewidth=1,linestyle='--',label='Medium Risk')
ax.legend(fontsize=9)
ax.set_xlabel('Fraud Susceptibility Index (0-100)')
ax.set_title('Which States Are Most Vulnerable to Medicaid Drug Fraud?\n(Composite: Spending x Price Variance x Suppression Rate)')
ax.grid(axis='x',alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD | Composite metric not official fraud measure',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'09-fraud-index.png')
print("[10] Pareto Concentration...")
ds10=drug_totals.sort_values('total_spending',ascending=False).reset_index(drop=True)
total_all=ds10['total_spending'].sum()
ds10['cum_pct']=ds10['total_spending'].cumsum()/total_all*100
ds10['drug_pct']=(ds10.index+1)/len(ds10)*100
top10_pct=ds10[ds10.index<10]['total_spending'].sum()/total_all*100
top20_pct=ds10[ds10.index<20]['total_spending'].sum()/total_all*100
fig,ax=plt.subplots(figsize=(12,8))
ax.fill_between(ds10['drug_pct'],ds10['cum_pct'],alpha=0.2,color=C_MID)
ax.plot(ds10['drug_pct'],ds10['cum_pct'],color=C_BLACK,linewidth=2.5,label='Actual')
ax.plot([0,100],[0,100],'--',color=C_LIGHT,linewidth=1.5,label='Perfect equality')
ax.axhline(y=top10_pct,color=C_DARK,linewidth=1,linestyle=':',alpha=0.7)
ax.annotate('Top 10 drugs = {:.1f}%'.format(top10_pct),xy=(10/len(ds10)*100,top10_pct),
    xytext=(25,top10_pct-12),fontsize=10,color=C_DARK,arrowprops=dict(arrowstyle='->',color=C_DARK,lw=1.5))
ax.annotate('Top 20 drugs = {:.1f}%'.format(top20_pct),xy=(20/len(ds10)*100,top20_pct),
    xytext=(40,top20_pct-8),fontsize=10,color=C_MID,arrowprops=dict(arrowstyle='->',color=C_MID,lw=1.5))
ax.set_xlabel('Cumulative % of Drug Products'); ax.set_ylabel('Cumulative % of Spending')
ax.set_title('The Medicaid Concentration Problem: Top 10 Drugs Consume {:.0f}% of Spending\n(Pareto Analysis, Sample 2018-2024)'.format(top10_pct))
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xlim(0,100); ax.set_ylim(0,105)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:'{:.0f}%'.format(x)))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:'{:.0f}%'.format(x)))
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'10-pareto-concentration.png')
print("[11] Seasonal Patterns...")
seas=combined.groupby(['year','quarter'])['total_amount_reimbursed'].sum().reset_index()
seas['quarter']=seas['quarter'].astype(str)
spiv=seas.pivot(index='quarter',columns='year',values='total_amount_reimbursed')
fig,ax=plt.subplots(figsize=(12,7))
xlbls=['Q1 Jan-Mar','Q2 Apr-Jun','Q3 Jul-Sep','Q4 Oct-Dec']
gsh=[C_BLACK,'#222222','#333333',C_MID,'#777777',C_LIGHT,'#cccccc']
ls11=['-','--','-.',':','-.','--','-']
for i,(year,col) in enumerate(spiv.items()):
    vals=[col.get(q,np.nan) for q in ['1','2','3','4']]
    ax.plot(xlbls,vals,marker='o',color=gsh[i%len(gsh)],linewidth=1.8,markersize=5,
        linestyle=ls11[i%len(ls11)],label=str(year))
ax.set_xlabel('Quarter'); ax.set_ylabel('Total Drug Spending (Sample)')
ax.set_title('Seasonal Medicaid Spending: When Do Drug Costs Peak?\n(Quarterly Reimbursements by Year, 2018-2024 Sample)')
ax.legend(fontsize=10,title='Year'); ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:'${:.0f}M'.format(x/1e6)))
ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'11-seasonal-patterns.png')
print("[12] Price Changes...")
dyp=combined[combined['unit_price'].notna()&(combined['unit_price']>0)].copy()
dyp=dyp[dyp['unit_price']<dyp['unit_price'].quantile(0.95)]
dyp['product_name']=dyp['product_name'].str.strip()
ap=dyp.groupby(['product_name','year'])['unit_price'].median().reset_index()
pp=ap.pivot(index='product_name',columns='year',values='unit_price')
ya=sorted(pp.columns)
if len(ya)>=2:
    y1p,y2p=ya[0],ya[-1]
    pp['pct']=((pp[y2p]-pp[y1p])/pp[y1p])*100
    pp=pp.dropna(subset=['pct'])
    pp=pp[pp.index!='']
    ch=pd.concat([pp.nlargest(10,'pct'),pp.nsmallest(8,'pct')]).drop_duplicates()
    ch=ch.sort_values('pct',ascending=True)
    fig,ax=plt.subplots(figsize=(12,9))
    clr12=[C_BLACK if v>50 else C_DARK if v>0 else C_LIGHT for v in ch['pct'].values]
    ax.barh(range(len(ch)),ch['pct'].values,color=clr12,edgecolor=C_DARK,linewidth=0.4)
    ax.set_yticks(range(len(ch))); ax.set_yticklabels(ch.index.values,fontsize=8)
    ax.axvline(x=0,color=C_DARK,linewidth=1.5)
    for i,val in enumerate(ch['pct'].values):
        ax.text(val+1 if val>=0 else val-1,i,'{:+.0f}%'.format(val),va='center',fontsize=7,ha='left' if val>=0 else 'right')
    ax.set_xlabel('Unit Price Change (%)')
    ax.set_title('Drug Price Shock: Biggest Movers in Medicaid {}-{}\n(Median Unit Price Change, Top Increases + Decreases)'.format(y1p,y2p))
    ax.grid(axis='x',alpha=0.3)
    ax.text(0.01,0.01,'Source: CMS SDUD via data.medicaid.gov',transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
    plt.tight_layout(); save_chart(fig,'12-price-changes.png')
print("[13] Suppression Rate...")
supp=combined[combined['state'].isin(VALID_STATES)].copy()
supp['suppressed']=(supp['suppression_used'].astype(str).str.lower()=='true').astype(int)
ss=supp.groupby('state').agg(total=('suppressed','count'),sup_count=('suppressed','sum')).reset_index()
ss['rate']=ss['sup_count']/ss['total']*100
ss=ss[ss['total']>50].sort_values('rate',ascending=True)
fig,ax=plt.subplots(figsize=(12,10))
n13=len(ss)
clr13=[C_BLACK if v>30 else C_DARK if v>15 else C_MID if v>5 else C_LIGHT for v in ss['rate'].values]
ax.barh(range(n13),ss['rate'].values,color=clr13,edgecolor=C_DARK,linewidth=0.4)
ax.set_yticks(range(n13)); ax.set_yticklabels(ss['state'].values,fontsize=8)
ax.set_xlabel('Data Suppression Rate (%)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:'{:.0f}%'.format(x)))
ax.set_title('What Are States Hiding? Medicaid Data Suppression Rates\n(% of Drug Records Suppressed — Higher = Less Transparency)')
ax.grid(axis='x',alpha=0.3)
ax.text(0.01,0.01,'Source: CMS SDUD suppression_used field | Privacy rules mask small-count records',
    transform=ax.transAxes,fontsize=8,color=C_LIGHT,va='bottom')
plt.tight_layout(); save_chart(fig,'13-suppression-rate.png')
print()
print("="*55)
print("SUMMARY STATISTICS")
print("="*55)
total_spend = combined['total_amount_reimbursed'].sum()
total_rxs   = combined['number_of_prescriptions'].sum()
opioid_spend= combined[combined['is_opioid']]['total_amount_reimbursed'].sum()
print("Sample records:     {:,}".format(len(combined)))
print("Sample spending:    ${:.3f}B".format(total_spend/1e9))
print("Sample Rx count:    {:.2f}M".format(total_rxs/1e6))
print("Opioid spending:    ${:.1f}M = {:.3f}%".format(opioid_spend/1e6, opioid_spend/total_spend*100))
print("Top10 concentration:{:.1f}%".format(top10_pct))
print("Top20 concentration:{:.1f}%".format(top20_pct))
print("Est growth rate:    +${:.1f}B/year".format(z_coef[0]))
print("Brand/generic ratio:{:.0f}x".format(r06))
print()
print("Top 5 drugs by spending:")
for _,row in drug_totals.head(5).iterrows():
    print("  {}: ${:.1f}M".format(row["product_name"], row["total_spending"]/1e6))
print()
print("Top 5 fraud-susceptible states:")
for _,row in fm.sort_values('fraud_idx',ascending=False).head(5).iterrows():
    print("  {}: {:.1f}/100".format(row["state"],row["fraud_idx"]))
print()
print("Top 5 suppression states:")
for _,row in ss.sort_values('rate',ascending=False).head(5).iterrows():
    print("  {}: {:.1f}%".format(row["state"],row["rate"]))
print()
print("Spending by year (estimated, $B):")
for y,b in zip(years_list,est_billions):
    print("  {}: ${:.2f}B".format(y,b))
print()
charts=sorted(os.listdir(CHARTS_DIR))
print("DONE. {} charts generated:".format(len(charts)))
for c in charts: print('  '+c)
