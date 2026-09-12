from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from .pipeline import choose_path, read_table, infer_map, build_supervised, add_features, make_preprocessor, audit, MODEL_DIR, REPORT_DIR
warnings.filterwarnings('ignore')

def split(df):
    n=len(df); a=max(1,int(n*.70)); b=max(a+1,int(n*.85)); return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()

def choose_threshold(y,p):
    best=(0.0,.5)
    for t in np.linspace(.05,.95,181):
        s=f1_score(y,(p>=t).astype(int),zero_division=0)
        if s>best[0]: best=(s,t)
    return float(best[1])

def eval_cls(y,p,t):
    pred=(p>=t).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'confusion_matrix':confusion_matrix(y,pred).tolist(),'threshold':float(t)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',choices=['auto','kaggle','demo'],default='auto'); args=ap.parse_args()
    path=choose_path(args.source); raw=read_table(path); m=infer_map(raw); x=build_supervised(raw,m); x,num,cat,features=add_features(x,m)
    if len(x)<100: raise ValueError(f'Only {len(x)} usable rows after cleaning; need a larger dataset.')
    audit_out=audit(raw,m); audit_out['usable_rows']=len(x); audit_out['late_rate']=float(x.late_payment.mean()); audit_out['features']=features
    (REPORT_DIR/'data_audit.json').write_text(json.dumps(audit_out,indent=2,default=str),encoding='utf8')
    tr,va,te=split(x)
    Xtr,ytr=tr[features],tr.late_payment; Xv,yv=va[features],va.late_payment; Xt,yt=te[features],te.late_payment
    prep=make_preprocessor(num,cat)
    models={
      'logistic_regression':Pipeline([('prep',prep),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]),
      'random_forest':Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1))]),
      'mlp':Pipeline([('prep',prep),('model',MLPClassifier(hidden_layer_sizes=(128,64,32),alpha=1e-4,learning_rate_init=7e-4,max_iter=600,early_stopping=True,random_state=42))]),
    }
    try:
      from xgboost import XGBClassifier
      pos=(ytr==1).sum(); neg=(ytr==0).sum(); sp=float(neg/max(pos,1))
      models['xgboost']=Pipeline([('prep',prep),('model',XGBClassifier(n_estimators=900,max_depth=5,learning_rate=.03,subsample=.9,colsample_bytree=.9,min_child_weight=2,reg_alpha=.03,reg_lambda=2.0,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=42,n_jobs=-1,scale_pos_weight=sp))])
    except Exception as e: print('XGBoost unavailable:',e)
    rows=[]; fitted={}
    for name,model in models.items():
      model.fit(Xtr,ytr); pv=model.predict_proba(Xv)[:,1]; th=choose_threshold(yv,pv); mv=eval_cls(yv,pv,th); rows.append({'model':name,'split':'validation',**mv}); fitted[name]=(model,th)
    comp=pd.DataFrame(rows).sort_values(['f1','recall','roc_auc'],ascending=False); best=str(comp.iloc[0].model); model,th=fitted[best]
    # Fit on train+validation only after selecting threshold/model.
    tv=pd.concat([tr,va],ignore_index=True); model.fit(tv[features],tv.late_payment); pte=model.predict_proba(Xt)[:,1]; tm=eval_cls(yt,pte,th)
    delayed_tv=tv[tv.late_payment==1]; delayed_te=te[te.late_payment==1]
    reg=None; rm={'skipped':True}
    if len(delayed_tv)>=20 and len(delayed_te)>=5:
      reg=Pipeline([('prep',make_preprocessor(num,cat)),('model',HistGradientBoostingRegressor(max_iter=400,learning_rate=.04,max_leaf_nodes=31,l2_regularization=.25,random_state=42))])
      reg.fit(delayed_tv[features],delayed_tv.delay_days_positive); pr=np.clip(reg.predict(delayed_te[features]),0,None)
      rm={'mae_days':float(mean_absolute_error(delayed_te.delay_days_positive,pr)),'rmse_days':float(np.sqrt(mean_squared_error(delayed_te.delay_days_positive,pr))),'n_test_delayed':int(len(delayed_te))}
    pred=pd.DataFrame({'late_probability':pte,'predicted_late':(pte>=th).astype(int),'actual_late':yt.to_numpy()})
    pred['risk_score']=np.round(pte*100,2); pred['estimated_exposure']=x.loc[te.index,'invoice_amount_clean'].to_numpy()*pte; pred['expected_delay_days']=np.clip(reg.predict(te[features]),0,None) if reg else 0.
    pred.to_csv(REPORT_DIR/'test_predictions.csv',index=False)
    comp.to_csv(REPORT_DIR/'model_comparison.csv',index=False)
    (REPORT_DIR/'test_metrics.json').write_text(json.dumps(tm,indent=2,default=str),encoding='utf8'); (REPORT_DIR/'regression_metrics.json').write_text(json.dumps(rm,indent=2,default=str),encoding='utf8')
    meta={'source_file':str(path),'column_map':m.__dict__,'features':features,'numeric_features':num,'categorical_features':cat,'best_model':best,'threshold':th,'test_metrics':tm,'regression_metrics':rm,'cold_start':'Prior customer features are constructed only from earlier invoices via shift(1). New customers have zero prior-history features.'}
    (MODEL_DIR/'metadata.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf8'); joblib.dump(model,MODEL_DIR/'classifier.joblib')
    if reg: joblib.dump(reg,MODEL_DIR/'delay_regressor.joblib')
    print(json.dumps({'best_model':best,'test_metrics':tm,'regression_metrics':rm},indent=2))
    if tm['accuracy']>=.90: print('PASS: test accuracy >= 90% on this dataset/split')
    else: print('NOTE: test accuracy < 90%. Do not claim otherwise; inspect audit and model comparison.')

if __name__=='__main__': main()
