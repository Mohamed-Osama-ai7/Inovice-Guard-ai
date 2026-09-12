from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.pipeline import choose_path, read_table, infer_map, build_supervised, add_features, make_preprocessor

ROOT=Path(__file__).resolve().parents[1]; MD=ROOT/'models'; RP=ROOT/'reports'; MD.mkdir(exist_ok=True); RP.mkdir(exist_ok=True)

def split(df):
    n=len(df); a=int(n*.70); b=int(n*.85); return df.iloc[:a],df.iloc[a:b],df.iloc[b:]

def cls_metrics(y,p,t=.5):
    pred=(p>=t).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p))}

def build_nlp_demo():
    # Controlled, diverse payment-communication examples for an optional NLP component.
    texts_pos=[
      'We expect a short delay while the payment receives internal approval.',
      'Our finance team will process the transfer next week, after approval.',
      'Payment may be delayed because the invoice is still under review.',
      'We are waiting for treasury approval before releasing the payment.',
      'The transfer is scheduled after our internal reconciliation is completed.',
      'We anticipate paying shortly, but the current processing cycle may take longer.',
      'There is a temporary cash flow constraint and the payment will be late.',
      'We cannot complete the payment by the due date; we expect to settle it soon.',
      'The remittance is pending and may miss the agreed payment date.',
      'Our accounting department requested additional time to complete payment.'
    ]
    texts_neg=[
      'Payment has been scheduled and will be completed on the agreed date.',
      'The invoice was approved and the bank transfer is ready for release.',
      'We have completed the payment and attached the remittance advice.',
      'Funds have been transferred successfully for this invoice.',
      'Our finance team confirmed payment for the original due date.',
      'The invoice is approved with no expected payment issues.',
      'Remittance advice is available; the transfer will settle as planned.',
      'We have no changes to the agreed payment schedule.',
      'Payment processing is complete and the transaction is confirmed.',
      'The outstanding invoice will be paid according to the contract terms.'
    ]
    # Repeat with light lexical perturbation to create a useful demo corpus without depending on one exact phrase.
    variants=['today','tomorrow','this week','this month','as agreed','after approval','after reconciliation']
    rows=[]
    for y,base in [(1,texts_pos),(0,texts_neg)]:
      for i,t in enumerate(base):
        for v in variants:
          rows.append((f'{t} We expect completion {v}.',y))
    df=pd.DataFrame(rows,columns=['text','label']).sample(frac=1,random_state=42).reset_index(drop=True)
    split_i=int(len(df)*.8); tr,te=df.iloc[:split_i],df.iloc[split_i:]
    pipe=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=5000)),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))])
    pipe.fit(tr.text,tr.label); p=pipe.predict_proba(te.text)[:,1]; met=cls_metrics(te.label.to_numpy(),p,.5)
    joblib.dump(pipe,MD/'nlp_payment_risk.joblib')
    (RP/'nlp_metrics.json').write_text(json.dumps({**met,'samples':len(df),'purpose':'Optional payment-communication risk signal demo model. Replace/augment with real annotated communications before production use.'},indent=2))


def build_tabular_variants():
    p=choose_path('demo'); raw=read_table(p); m=infer_map(raw); x=build_supervised(raw,m); x,num,cat,features=add_features(x,m)
    tr,va,te=split(x); Xtr,ytr=tr[features],tr.late_payment; Xv,yv=va[features],va.late_payment; Xt,yt=te[features],te.late_payment
    models={
      'logistic_regression':Pipeline([('prep',make_preprocessor(num,cat)),('model',LogisticRegression(max_iter=3000,class_weight='balanced'))]),
      'random_forest':Pipeline([('prep',make_preprocessor(num,cat)),('model',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1))]),
      'mlp_neural_network':Pipeline([('prep',make_preprocessor(num,cat)),('model',MLPClassifier(hidden_layer_sizes=(128,64,32),alpha=1e-4,learning_rate_init=7e-4,max_iter=600,early_stopping=True,random_state=42))])
    }
    out={}
    for name,mod in models.items():
      mod.fit(Xtr,ytr); p=mod.predict_proba(Xv)[:,1]
      met=cls_metrics(yv,p,.5); out[name]=met
      joblib.dump(mod,MD/f'{name}.joblib')
    (RP/'all_model_artifacts.json').write_text(json.dumps({'source':str(p),'saved_models':sorted(out),'validation_metrics':out},indent=2))

if __name__=='__main__':
    build_tabular_variants(); build_nlp_demo(); print('Built ML baselines, MLP, and NLP artifacts.')
