# -*- coding: utf-8 -*-
"""
MODELO DE SATISFACCION DEL VISITANTE - Puno / Lago Titicaca
Dataset: 500 resenas reales de Google Maps (Outscraper) -> filtro 2020-2025
Metodo tipo 'anemia': features -> clasificacion (RF/XGBoost) + SHAP, + modelo de texto (TF-IDF).
"""
import sys, io, os, json, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_score, recall_score)
import xgboost as xgb
import shap

SRC  = sys.argv[1]
OUT  = r"C:\Users\HP\Documents\Modelo_Satisfaccion_Titicaca"
FIG  = os.path.join(OUT, "figuras")
os.makedirs(FIG, exist_ok=True)

# ---- estilo grafico (limpio, accesible) ----
PAL = {'teal':'#0E7C7B','orange':'#E8833A','gray':'#6B7280','green':'#2E8B57',
       'red':'#C0392B','blue':'#2C6E9B','light':'#CBD5E1'}
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':150,'font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold','axes.spines.top':False,
    'axes.spines.right':False,'font.family':'DejaVu Sans','axes.grid':True,
    'grid.alpha':0.25,'grid.linestyle':'--'})

print("="*70); print("  MODELO DE SATISFACCION - PUNO / LAGO TITICACA"); print("="*70)

# ============ 1. CARGA Y LIMPIEZA ============
df = pd.read_excel(SRC, sheet_name=0)
# nombres de lugar legibles y consistentes con el estudio
place_map = {
 'Islas de Los Uros':'Islas Uros','Taquile':'Isla Taquile','Amantaní':'Isla Amantaní',
 'Reserva Nacional Titicaca':'Reserva Nac. Titicaca','Sillustani':'Sillustani'}
df['lugar'] = df['name'].map(lambda x: place_map.get(str(x).strip(), str(x).strip()))
df['fecha'] = pd.to_datetime(df['review_datetime_utc'], errors='coerce', utc=True)
df['anio']  = df['fecha'].dt.year
df['mes']   = df['fecha'].dt.month
df['estrellas'] = pd.to_numeric(df['review_rating'], errors='coerce')
df['texto'] = (df['review_text'].fillna('').astype(str)
    .str.replace('\u200b','',regex=False)
    .str.replace(r'<[^>]+>',' ',regex=True)        # quita HTML tipo <br>
    .str.replace(r'https?://\S+',' ',regex=True)   # quita URLs
    .str.strip())
df.loc[df['texto'].str.lower()=='nan','texto'] = ''

n0 = len(df)
# FILTRO AL PERIODO DEL ESTUDIO 2020-2025 (coincide con el titulo)
df = df[(df['anio']>=2020)&(df['anio']<=2025)].copy()
df = df.dropna(subset=['estrellas']).reset_index(drop=True)
print(f"\nResenas totales descargadas : {n0}")
print(f"Resenas en periodo 2020-2025: {len(df)}")

# ============ 2. VARIABLE OBJETIVO (satisfaccion) ============
# Satisfecho = 4 o 5 estrellas  |  Insatisfecho = 1-3
df['satisfecho'] = (df['estrellas']>=4).astype(int)
tasa = df['satisfecho'].mean()
print(f"\nTASA DE SATISFACCION GLOBAL: {tasa*100:.1f}%  "
      f"(satisfechos={df['satisfecho'].sum()}, insatisfechos={(1-df['satisfecho']).sum()})")

# ============ 3. INGENIERIA DE VARIABLES (features) ============
SP_POS = set("""bueno buena excelente hermoso hermosa bonito bonita maravilloso increible
genial recomiendo recomendable encanto encanta espectacular linda lindo perfecto tranquilo
amable amables limpio impresionante fascinante magico unico paz paisaje vista""".split())
SP_NEG = set("""malo mala pesimo caro cara sucio sucia feo fea decepcion decepcionante estafa
nada aburrido demora espera problema maltrato basura horrible desorden caos frio incomodo
tristeza pobre explotacion lamentable no""".split())

def feat_text(t):
    t=str(t); low=t.lower(); words=re.findall(r"\w+", low)
    nw=len(words); letters=[c for c in t if c.isalpha()]
    up=sum(1 for c in letters if c.isupper())/max(1,len(letters))
    pos=sum(1 for w in words if w in SP_POS); neg=sum(1 for w in words if w in SP_NEG)
    return pd.Series({'n_palabras':nw,'n_caracteres':len(t),'exclamaciones':t.count('!'),
        'interrogaciones':t.count('?'),'ratio_mayus':round(up,3),
        'palabras_positivas':pos,'palabras_negativas':neg,'polaridad_lexico':pos-neg})
ft = df['texto'].apply(feat_text)
df = pd.concat([df, ft], axis=1)

# idioma aproximado (es / en / otro) por solapamiento de stopwords -> apoya enfoque multilingue
ES={'de','la','el','que','y','en','un','una','por','con','para','los','las','muy','es','lugar'}
EN={'the','and','a','to','of','is','in','it','for','very','place','this','was','with','you'}
def lang(t):
    w=set(re.findall(r"\w+", str(t).lower()))
    es=len(w&ES); en=len(w&EN)
    if es==0 and en==0: return 'otro'
    return 'es' if es>=en else ('en' if en>es else 'otro')
df['idioma']=df['texto'].apply(lang)

df['temporada_alta'] = df['mes'].isin([5,6,7,8,9,10]).astype(int)   # seca, alta en Puno
df['candelaria']     = (df['mes']==2).astype(int)                    # Fiesta Candelaria
for c in ['author_reviews_count','author_photos_count','author_ratings_count','review_likes']:
    df[c]=pd.to_numeric(df.get(c), errors='coerce').fillna(0)
df['tiene_foto'] = df['review_img_url'].notna().astype(int) if 'review_img_url' in df else 0

# matriz de features (interpretables, estilo anemia)
feat_cols = ['lugar','anio','mes','temporada_alta','candelaria','idioma',
    'n_palabras','n_caracteres','exclamaciones','interrogaciones','ratio_mayus',
    'palabras_positivas','palabras_negativas','polaridad_lexico',
    'tiene_foto','author_reviews_count','author_photos_count','review_likes']
X = pd.get_dummies(df[feat_cols], columns=['lugar','idioma'], drop_first=False)
X.columns = [c.replace('lugar_','').replace('idioma_','idioma=') for c in X.columns]
y = df['satisfecho'].values

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
print(f"\nEntrenamiento: {len(Xtr)}  |  Prueba: {len(Xte)}  |  Features: {X.shape[1]}")

# ============ 4. MODELOS (contexto/metadatos) ============
spw = (y==0).sum()/max(1,(y==1).sum())   # balance para xgboost (clase positiva=satisfecho)
modelos = {
 'Random Forest': RandomForestClassifier(n_estimators=400,max_depth=None,
        class_weight='balanced',random_state=42,n_jobs=-1),
 'XGBoost': xgb.XGBClassifier(n_estimators=350,max_depth=4,learning_rate=0.07,
        subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',
        scale_pos_weight=spw,random_state=42),
}
res={}
for nm,mdl in modelos.items():
    mdl.fit(Xtr,ytr)
    p=mdl.predict(Xte); pr=mdl.predict_proba(Xte)[:,1]
    res[nm]=dict(acc=accuracy_score(yte,p),auc=roc_auc_score(yte,pr),
        f1=f1_score(yte,p),f1m=f1_score(yte,p,average='macro'),
        prec=precision_score(yte,p,zero_division=0),rec=recall_score(yte,p),
        proba=pr,pred=p,model=mdl)
    print(f"\n[{nm}]  Acc={res[nm]['acc']:.3f}  AUC={res[nm]['auc']:.3f}  "
          f"F1(satisf)={res[nm]['f1']:.3f}  F1-macro={res[nm]['f1m']:.3f}")

# ============ 5. MODELO DE TEXTO (TF-IDF + Reg. Logistica) ============
# stopwords multilingues (es/en/fr/pt/it) + fragmentos frecuentes
SW_EXTRA = {'muy','mas','pero','todo','nos','fue','asi','ahi','ya','sin','su','se','lo',
   'al','le','me','mi','si','son','esta','este','tambien','porque','como','donde','cuando',
   'hay','ser','hace','hacer','puede','bien','ver','tiene','sus','una','uno','del','los','las',
   # frances/portugues/italiano/ingles que se colaron
   'et','je','tu','il','elle','nous','vous','ils','est','pas','pour','avec','sur','une','des',
   'les','dans','que','qui','sao','nao','para','com','mais','muito','tem','por','uma','seu',
   'che','per','con','una','gli','sono','anche','the','and','for','you','was','are','very',
   'this','that','with','have','but','not','they','there','here','its','our','your','from',
   'br','http','https','www','com','good','nice','place','great'}
SW = list(ES|EN|SW_EXTRA)
tdf = df[df['texto'].str.len()>0].copy()
Xt_tr,Xt_te,yt_tr,yt_te = train_test_split(
    tdf['texto'], tdf['satisfecho'], test_size=0.30,random_state=42,stratify=tdf['satisfecho'])
txt_model = Pipeline([
  ('tfidf',TfidfVectorizer(strip_accents='unicode',lowercase=True,ngram_range=(1,2),
        min_df=3,max_df=0.9,stop_words=SW,sublinear_tf=True,
        token_pattern=r"(?u)\b[a-záéíóúñü]{2,}\b")),   # 2+ letras (conserva 'no')
  ('clf',LogisticRegression(max_iter=2000,class_weight='balanced',C=3.0))])
txt_model.fit(Xt_tr,yt_tr)
pt=txt_model.predict(Xt_te); prt=txt_model.predict_proba(Xt_te)[:,1]
res['Texto (TF-IDF)']=dict(acc=accuracy_score(yt_te,pt),auc=roc_auc_score(yt_te,prt),
   f1=f1_score(yt_te,pt),f1m=f1_score(yt_te,pt,average='macro'),
   prec=precision_score(yt_te,pt,zero_division=0),rec=recall_score(yt_te,pt))
print(f"\n[Texto TF-IDF] Acc={res['Texto (TF-IDF)']['acc']:.3f}  "
      f"AUC={res['Texto (TF-IDF)']['auc']:.3f}  F1(satisf)={res['Texto (TF-IDF)']['f1']:.3f}")

# palabras que mas empujan satisfaccion / insatisfaccion
vec=txt_model.named_steps['tfidf']; coef=txt_model.named_steps['clf'].coef_[0]
voc=np.array(vec.get_feature_names_out()); order=np.argsort(coef)
# para MOSTRAR: filtra fragmentos de otros idiomas (el modelo igual los usa)
FOREIGN={'der','die','und','ist','sua','seu','nao','sao','piu','che','gli','sono','est','et',
   'je','tu','il','pas','pour','avec','les','des','dans','interessant','interessante','tres',
   'muito','para','com','mais','the','and','for','very','nice','good','great','place','isla'}
def pick(idxs,n=15):
    out=[]
    for i in idxs:
        w=voc[i]
        if any(t in FOREIGN for t in w.split()): continue
        out.append((w,coef[i]))
        if len(out)>=n: break
    return out
top_pos=pick(order[::-1]); top_neg=pick(order)
print("\nPalabras PRO-satisfaccion:", ", ".join(w for w,_ in top_pos[:10]))
print("Palabras PRO-insatisfaccion:", ", ".join(w for w,_ in top_neg[:10]))

# mejor modelo por AUC (entre RF/XGB para SHAP)
best_name=max(['Random Forest','XGBoost'], key=lambda k:res[k]['auc'])
best=res[best_name]['model']
print(f"\n>> Mejor modelo tabular (SHAP): {best_name}")
print("\n--- Reporte de clasificacion (mejor tabular) ---")
print(classification_report(yte,res[best_name]['pred'],
      target_names=['Insatisfecho','Satisfecho'],zero_division=0))

# ============ 6. GRAFICOS ============
# 6.1 distribucion de estrellas + torta satisfaccion
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
vc=df['estrellas'].value_counts().sort_index()
cols=[PAL['red'] if s<=3 else PAL['green'] for s in vc.index]
ax[0].bar(vc.index.astype(int).astype(str),vc.values,color=cols,edgecolor='white')
ax[0].set_title('Distribución de calificaciones (estrellas)');ax[0].set_xlabel('Estrellas');ax[0].set_ylabel('N.º de reseñas')
for i,v in zip(vc.index.astype(int).astype(str),vc.values): ax[0].text(i,v+1,str(v),ha='center',fontsize=9)
sat=[df['satisfecho'].sum(),(1-df['satisfecho']).sum()]
ax[1].pie(sat,labels=[f'Satisfecho\n{sat[0]} ({sat[0]/len(df)*100:.0f}%)',
    f'Insatisfecho\n{sat[1]} ({sat[1]/len(df)*100:.0f}%)'],colors=[PAL['green'],PAL['red']],
    autopct='',startangle=90,wedgeprops=dict(edgecolor='white',width=0.45))
ax[1].set_title('Satisfacción del visitante (⭐4-5 = satisfecho)')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig1_satisfaccion.png'),bbox_inches='tight');plt.close()

# 6.2 tasa de satisfaccion por atractivo
g=df.groupby('lugar')['satisfecho'].agg(['mean','count']).sort_values('mean')
fig,ax=plt.subplots(figsize=(8,4.2))
b=ax.barh(g.index,g['mean']*100,color=PAL['teal'],edgecolor='white')
for i,(m,c) in enumerate(zip(g['mean']*100,g['count'])): ax.text(m+0.6,i,f'{m:.0f}%  (n={c})',va='center',fontsize=9)
ax.set_xlim(0,105);ax.set_xlabel('% de visitantes satisfechos');ax.set_title('Satisfacción por atractivo turístico — Titicaca 2020-2025')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig2_por_atractivo.png'),bbox_inches='tight');plt.close()

# 6.3 evolucion por anio
gy=df.groupby('anio')['satisfecho'].agg(['mean','count'])
fig,ax=plt.subplots(figsize=(8,4.2));ax2=ax.twinx()
ax.bar(gy.index.astype(int).astype(str),gy['count'],color=PAL['light'],label='N.º reseñas')
ax2.plot(gy.index.astype(int).astype(str),gy['mean']*100,'-o',color=PAL['orange'],lw=2.5,label='% satisfechos')
ax.set_ylabel('N.º de reseñas');ax2.set_ylabel('% satisfechos');ax2.set_ylim(0,105)
ax.set_title('Volumen y satisfacción por año (2020-2025)');ax.set_xlabel('Año')
for x,m in zip(gy.index.astype(int).astype(str),gy['mean']*100): ax2.text(x,m+2,f'{m:.0f}%',ha='center',color=PAL['orange'],fontsize=9,fontweight='bold')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig3_por_anio.png'),bbox_inches='tight');plt.close()

# 6.4 ROC comparativo
fig,ax=plt.subplots(figsize=(6,5.2))
for nm,cl in [('Random Forest',PAL['teal']),('XGBoost',PAL['orange'])]:
    fpr,tpr,_=roc_curve(yte,res[nm]['proba']);ax.plot(fpr,tpr,color=cl,lw=2,label=f"{nm} (AUC={res[nm]['auc']:.2f})")
fpr,tpr,_=roc_curve(yt_te,prt);ax.plot(fpr,tpr,color=PAL['blue'],lw=2,label=f"Texto TF-IDF (AUC={res['Texto (TF-IDF)']['auc']:.2f})")
ax.plot([0,1],[0,1],'--',color=PAL['gray']);ax.set_xlabel('Falsos positivos');ax.set_ylabel('Verdaderos positivos')
ax.set_title('Curvas ROC — predicción de satisfacción');ax.legend(loc='lower right',fontsize=9)
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig4_roc.png'),bbox_inches='tight');plt.close()

# 6.5 matriz de confusion (mejor tabular)
cm=confusion_matrix(yte,res[best_name]['pred'])
fig,ax=plt.subplots(figsize=(4.6,4.2))
im=ax.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=14,
        color='white' if cm[i,j]>cm.max()/2 else 'black')
ax.set_xticks([0,1],['Insatisf.','Satisf.']);ax.set_yticks([0,1],['Insatisf.','Satisf.'])
ax.set_xlabel('Predicho');ax.set_ylabel('Real');ax.set_title(f'Matriz de confusión — {best_name}')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig5_confusion.png'),bbox_inches='tight');plt.close()

# 6.6 palabras top (texto)
fig,ax=plt.subplots(1,2,figsize=(11,5))
pw=top_pos[::-1];nw=top_neg[::-1]
ax[0].barh([w for w,_ in pw],[c for _,c in pw],color=PAL['green']);ax[0].set_title('Palabras que predicen SATISFACCIÓN')
ax[1].barh([w for w,_ in nw],[abs(c) for _,c in nw],color=PAL['red']);ax[1].set_title('Palabras que predicen INSATISFACCIÓN')
for a in ax: a.set_xlabel('Peso en el modelo (|coef|)')
plt.tight_layout();plt.savefig(os.path.join(FIG,'fig6_palabras.png'),bbox_inches='tight');plt.close()

# ============ 7. SHAP (mejor modelo tabular) ============
try:
    expl=shap.TreeExplainer(best)
    sv=expl.shap_values(Xte)
    if isinstance(sv,list): sv=sv[1]
    # beeswarm
    plt.figure();shap.summary_plot(sv,Xte,show=False,max_display=14,plot_size=(9,6))
    plt.title(f'SHAP — factores de la satisfacción ({best_name})',fontsize=12,fontweight='bold')
    plt.tight_layout();plt.savefig(os.path.join(FIG,'fig7_shap_beeswarm.png'),bbox_inches='tight');plt.close()
    # barras importancia media
    plt.figure();shap.summary_plot(sv,Xte,plot_type='bar',show=False,max_display=14,plot_size=(9,6),color=PAL['teal'])
    plt.title('SHAP — importancia media de cada factor',fontsize=12,fontweight='bold')
    plt.tight_layout();plt.savefig(os.path.join(FIG,'fig8_shap_barras.png'),bbox_inches='tight');plt.close()
    mean_abs=np.abs(sv).mean(0); imp=pd.Series(mean_abs,index=Xte.columns).sort_values(ascending=False)
    print("\n--- TOP 10 factores por SHAP ---");print(imp.head(10).round(4).to_string())
    shap_ok=True
except Exception as e:
    print("SHAP fallo:",e);shap_ok=False;imp=pd.Series(best.feature_importances_,index=X.columns).sort_values(ascending=False)

# ============ 8. GUARDAR RESULTADOS ============
keep=['lugar','anio','mes','estrellas','satisfecho','idioma','n_palabras','polaridad_lexico',
      'palabras_positivas','palabras_negativas','temporada_alta','review_likes','texto']
df[keep].to_csv(os.path.join(OUT,'01_reviews_limpias_2020-2025.csv'),index=False,encoding='utf-8-sig')

tabla=pd.DataFrame({k:{'Accuracy':round(v['acc'],3),'AUC':round(v['auc'],3),
   'F1 (satisf.)':round(v['f1'],3),'F1-macro':round(v['f1m'],3),
   'Precisión':round(v['prec'],3),'Recall':round(v['rec'],3)} for k,v in res.items()}).T
tabla.to_csv(os.path.join(OUT,'02_metricas_modelos.csv'),encoding='utf-8-sig')
print("\n================ TABLA DE MÉTRICAS ================")
print(tabla.to_string())

resumen={'n_total_descargadas':int(n0),'n_periodo_2020_2025':int(len(df)),
  'tasa_satisfaccion_%':round(tasa*100,1),
  'satisfechos':int(df['satisfecho'].sum()),'insatisfechos':int((1-df['satisfecho']).sum()),
  'mejor_modelo_tabular':best_name,'auc_mejor_tabular':round(res[best_name]['auc'],3),
  'auc_texto':round(res['Texto (TF-IDF)']['auc'],3),
  'metricas':{k:{kk:round(float(vv),3) for kk,vv in v.items() if kk in('acc','auc','f1','f1m','prec','rec')} for k,v in res.items()},
  'top_factores_shap':list(imp.head(10).index),
  'palabras_satisfaccion':[w for w,_ in top_pos],'palabras_insatisfaccion':[w for w,_ in top_neg],
  'atractivos':{k:{'tasa_%':round(v*100,1),'n':int(c)} for k,v,c in zip(g.index,g['mean'],g['count'])}}
with open(os.path.join(OUT,'03_resumen_resultados.json'),'w',encoding='utf-8') as f:
    json.dump(resumen,f,ensure_ascii=False,indent=2)

print("\n>>> GUARDADO EN:",OUT)
print(">>> Figuras:",len(os.listdir(FIG)),"PNG en",FIG)
print(">>> OK TERMINADO")
