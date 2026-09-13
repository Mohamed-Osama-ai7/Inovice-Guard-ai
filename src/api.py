from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.customer_360 import get_customer_profile

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'models'; TEMPLATES=ROOT/'templates'
app=FastAPI(title='InvoiceGuard AI',version='1.0.0')
classifier=joblib.load(MODELS/'classifier.joblib') if (MODELS/'classifier.joblib').exists() else None
regressor=joblib.load(MODELS/'delay_regressor.joblib') if (MODELS/'delay_regressor.joblib').exists() else None
meta=json.loads((MODELS/'metadata.json').read_text()) if (MODELS/'metadata.json').exists() else {}

class InvoiceInput(BaseModel):
    invoice_date: str
    due_date: str
    invoice_amount: float=Field(gt=0)
    customer_seen_before: int=0
    prior_late_count: float=0
    prior_late_ratio: float=0
    prior_avg_delay: float=0
    industry: Optional[str]=None
    company_size: Optional[str]=None
    payment_method: Optional[str]=None
    customer_segment: Optional[str]=None
    outstanding_amount: float=0

def row(i:InvoiceInput):
    inv=pd.Timestamp(i.invoice_date); due=pd.Timestamp(i.due_date)
    d={'invoice_amount_clean':i.invoice_amount,'amount_log1p':float(np.log1p(i.invoice_amount)),'days_to_due':(due-inv).days,'invoice_year':inv.year,'invoice_month':inv.month,'invoice_quarter':inv.quarter,'invoice_dayofweek':inv.dayofweek,'customer_seen_before':i.customer_seen_before,'customer_is_new':int(i.customer_seen_before==0),'prior_late_count':i.prior_late_count,'prior_late_ratio':i.prior_late_ratio,'prior_avg_delay':i.prior_avg_delay,'outstanding_amount':i.outstanding_amount}
    for c in ['industry','company_size','payment_method','customer_segment']: d[c]=getattr(i,c)
    return pd.DataFrame([d])

@app.get('/',response_class=HTMLResponse)
def home():
    return (TEMPLATES/'index.html').read_text(encoding='utf8')

@app.get('/health')
def health(): return {'status':'ok','model_loaded':classifier is not None,'best_model':meta.get('best_model'),'test_metrics':meta.get('test_metrics',{})}

@app.get('/metrics')
def metrics():
    return {'best_model':meta.get('best_model'),'test_metrics':meta.get('test_metrics',{}),'regression_metrics':meta.get('regression_metrics',{}),'data_source':meta.get('source_file')}

@app.post('/predict')
def predict(i:InvoiceInput):
    if classifier is None: return {'error':'Model not trained. Run: python -m src.train --source demo'}
    X=row(i); p=float(classifier.predict_proba(X)[:,1][0]); th=float(meta.get('threshold',.5)); late=int(p>=th)
    delay=float(np.clip(regressor.predict(X)[0],0,None)) if regressor else 0.0
    exposure=float(i.invoice_amount*p)
    risk='LOW' if p<.35 else ('MEDIUM' if p<.65 else ('HIGH' if p<.85 else 'CRITICAL'))
    reasons=[]
    if i.prior_late_ratio>=.5: reasons.append('High historical late-payment ratio')
    if i.prior_avg_delay>=10: reasons.append('High historical average delay')
    if i.outstanding_amount>i.invoice_amount: reasons.append('Outstanding balance exceeds current invoice amount')
    if i.invoice_amount>=100000: reasons.append('Large invoice amount')
    if i.customer_seen_before==0: reasons.append('New customer: prediction relies on profile/invoice features')
    rec='Prioritize collection follow-up before the due date.' if risk in ('HIGH','CRITICAL') else 'Monitor payment and schedule routine follow-up.'
    return {'late_probability':round(p,4),'predicted_late':late,'expected_delay_days':round(delay,1),'risk_level':risk,'estimated_financial_exposure':round(exposure,2),'explanation':reasons[:4],'recommendation':rec}

@app.get('/customer/profile/{customer_id}')
def customer_profile(customer_id: str):
    profile = get_customer_profile(customer_id)
    if not profile.get("data_sources"):
        return {"error": "Customer not found"}
    return profile
