# -*- coding: utf-8 -*-
"""MODELO DE DEMANDA (Prophet) - Flujo de visitantes al Lago Titicaca 2020-2025.
Serie mensual oficial MINCETUR -> pronostico 2026-2027 + validacion + estacionalidad."""
import sys, io, os, json, logging, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore'); logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prophet import Prophet

SRC=r"D:\descargas 2\dataset de islas puno\00_BaseDatos_CONSOLIDADA_Titicaca_2020-2025.xlsx"
OUT=r"C:\Users\HP\Documents\Modelo_Demanda_Titicaca"; FIG=os.path.join(OUT,"figuras")
os.makedirs(FIG, exist_ok=True)
PAL={'teal':'#0E7C7B','orange':'#E8833A','gray':'#6B7280','blue':'#2C6E9B','light':'#CBD5E1'}
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':150,'font.size':11,'axes.titlesize':13,
 'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
 'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--'})

# ---------- 1. cargar serie mensual ----------
raw=pd.read_excel(SRC, sheet_name='Consolidado mensual', header=2)
raw=raw.rename(columns={c:str(c).strip() for c in raw.columns})
raw=raw[pd.to_numeric(raw['Anio'],errors='coerce').notna()].copy()
raw['Anio']=raw['Anio'].astype(int)
MES={'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,'julio':7,'agosto':8,
     'septiembre':9,'setiembre':9,'octubre':10,'noviembre':11,'diciembre':12}
def pmes(v):
    s=str(v).strip().lower()
    if s in MES: return MES[s]
    try: return int(float(s))
    except: return None
raw['Mes']=raw['Mes'].apply(pmes)
raw=raw[raw['Mes'].notna()].copy(); raw['Mes']=raw['Mes'].astype(int)
raw['TOTAL Titicaca']=pd.to_numeric(raw['TOTAL Titicaca'],errors='coerce')
raw['ds']=pd.to_datetime(dict(year=raw['Anio'],month=raw['Mes'],day=1))
df=raw[['ds','TOTAL Titicaca']].rename(columns={'TOTAL Titicaca':'y'}).dropna().sort_values('ds').reset_index(drop=True)
print("Meses:",len(df),"| de",df['ds'].min().date(),"a",df['ds'].max().date())
print("Total por año:")
print(df.assign(anio=df['ds'].dt.year).groupby('anio')['y'].sum().to_string())

def make_model():
    return Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                   seasonality_mode='additive', interval_width=0.80, changepoint_prior_scale=0.05)

# ---------- 2. VALIDACION: entrena 2020-2024, predice 2025 ----------
train=df[df['ds'].dt.year<=2024]; test=df[df['ds'].dt.year==2025]
m0=make_model(); m0.fit(train)
fc0=m0.predict(test[['ds']].copy()); fc0['yhat']=fc0['yhat'].clip(lower=0)
y_true=test['y'].values; y_pred=fc0['yhat'].values
mae=np.mean(np.abs(y_true-y_pred)); rmse=np.sqrt(np.mean((y_true-y_pred)**2))
mape=np.mean(np.abs((y_true-y_pred)/np.maximum(y_true,1)))*100
print(f"\nVALIDACION 2025 -> MAE={mae:,.0f} RMSE={rmse:,.0f} MAPE={mape:.1f}%")

# ---------- 3. MODELO FINAL (toda la serie) + pronostico 24 meses ----------
m=make_model(); m.fit(df)
future=m.make_future_dataframe(periods=24, freq='MS')
fc=m.predict(future)
for c in ['yhat','yhat_lower','yhat_upper']: fc[c]=fc[c].clip(lower=0)
pred=fc[fc['ds']>df['ds'].max()]
anual_pred=pred.assign(anio=pred['ds'].dt.year).groupby('anio')['yhat'].sum()
print("\nPRONOSTICO anual (visitantes):")
print(anual_pred.round(0).astype(int).to_string())

# ---------- 4. GRAFICOS ----------
# 4.1 forecast historico + prediccion + banda
fig,ax=plt.subplots(figsize=(10,4.6))
hist=fc[fc['ds']<=df['ds'].max()]
ax.plot(df['ds'],df['y'],'o-',color=PAL['teal'],ms=3,lw=1.4,label='Histórico (real)')
ax.plot(pred['ds'],pred['yhat'],'-',color=PAL['orange'],lw=2.4,label='Pronóstico 2026–2027')
ax.fill_between(pred['ds'],pred['yhat_lower'],pred['yhat_upper'],color=PAL['orange'],alpha=0.18,label='Banda 80%')
ax.axvline(df['ds'].max(),color=PAL['gray'],ls='--',lw=1)
ax.set_title('Pronóstico de demanda — visitantes al Lago Titicaca (Prophet)')
ax.set_xlabel('Año'); ax.set_ylabel('Visitantes / mes'); ax.legend(loc='upper left',fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_demanda_forecast.png'),bbox_inches='tight'); plt.close()

# 4.2 validacion 2025
fig,ax=plt.subplots(figsize=(9,4.2))
ax.plot(train['ds'],train['y'],'o-',color=PAL['teal'],ms=3,lw=1.3,label='Entrenamiento 2020–2024')
ax.plot(test['ds'],y_true,'o-',color=PAL['blue'],ms=4,lw=1.8,label='Real 2025')
ax.plot(test['ds'],y_pred,'s--',color=PAL['orange'],ms=4,lw=1.8,label='Predicho 2025')
ax.set_title(f'Validación: predicción de 2025 (MAPE={mape:.0f}%)')
ax.set_xlabel('Año'); ax.set_ylabel('Visitantes / mes'); ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_demanda_validacion.png'),bbox_inches='tight'); plt.close()

# 4.3 estacionalidad mensual (del modelo)
meses=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
one_year=pd.DataFrame({'ds':pd.date_range('2024-01-01',periods=12,freq='MS')})
comp=m.predict(one_year); seas=comp['yearly'].values
fig,ax=plt.subplots(figsize=(9,4))
cols=[PAL['teal'] if v>=0 else PAL['gray'] for v in seas]
ax.bar(meses,seas,color=cols,edgecolor='white')
ax.axhline(0,color=PAL['gray'],lw=1)
ax.set_title('Estacionalidad: meses con mayor/menor demanda')
ax.set_ylabel('Efecto estacional (×)');
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_demanda_estacionalidad.png'),bbox_inches='tight'); plt.close()

# ---------- 5. GUARDAR ----------
df.assign(anio=df['ds'].dt.year).to_csv(os.path.join(OUT,'serie_mensual_titicaca.csv'),index=False,encoding='utf-8-sig')
pred[['ds','yhat','yhat_lower','yhat_upper']].to_csv(os.path.join(OUT,'pronostico_2026_2027.csv'),index=False,encoding='utf-8-sig')
top_mes=meses[int(np.argmax(seas))]; low_mes=meses[int(np.argmin(seas))]
resumen={'n_meses':int(len(df)),'periodo':f"{df['ds'].min().date()} a {df['ds'].max().date()}",
 'validacion_2025':{'MAE':round(float(mae),0),'RMSE':round(float(rmse),0),'MAPE_%':round(float(mape),1)},
 'pronostico_anual':{int(k):int(round(v)) for k,v in anual_pred.items()},
 'mes_pico':top_mes,'mes_bajo':low_mes,
 'total_2025_real':int(df[df['ds'].dt.year==2025]['y'].sum())}
with open(os.path.join(OUT,'resumen_demanda.json'),'w',encoding='utf-8') as f:
    json.dump(resumen,f,ensure_ascii=False,indent=2)
print("\nMes pico:",top_mes,"| Mes bajo:",low_mes)
print(">>> GUARDADO EN:",OUT,"| figuras:",len(os.listdir(FIG)))
print(">>> OK")
