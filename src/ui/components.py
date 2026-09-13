import streamlit as st
from typing import List, Dict, Any

def badge(label: str) -> str:
    """Return HTML for a semantic status badge."""
    lbl = str(label).upper()
    if lbl in ["LOW", "GOOD", "ACTIVE"]:
        c = "status-good"
    elif lbl in ["MEDIUM", "WARNING", "AT RISK"]:
        c = "status-warning"
    elif lbl in ["HIGH", "DANGER", "LATE"]:
        c = "status-danger"
    elif lbl in ["CRITICAL", "CHURNED"]:
        c = "status-critical"
    else:
        c = "status-info"
    return f'<span class="ig-badge {c}">{label}</span>'

def page_header(title: str, subtitle: str = "", right_content: str = "") -> None:
    """Render a professional enterprise page header."""
    html = f'''
    <div class="ig-page-header">
      <div class="ig-page-header-left">
        <h1 class="ig-page-header-title">{title}</h1>
        <p class="ig-page-header-subtitle">{subtitle}</p>
      </div>
      <div class="ig-page-header-right">
        {right_content}
      </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def section_label(text: str) -> None:
    st.markdown(f"### {text}", unsafe_allow_html=True)

def divider() -> None:
    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:2rem 0;'/>", unsafe_allow_html=True)

def kpi_row(items: List[Dict[str, Any]]) -> None:
    """
    Renders a row of KPI cards. 
    items is a list of dicts: {"icon": str, "label": str, "value": str, "trend": str, "trend_dir": "up"|"down"|"neutral"}
    """
    if not items:
        return
    
    cols = st.columns(len(items))
    for i, col in enumerate(cols):
        item = items[i]
        icon = item.get("icon", "")
        label = item.get("label", "")
        value = item.get("value", "")
        trend = item.get("trend", "")
        trend_dir = item.get("trend_dir", "neutral")
        
        trend_html = ""
        if trend:
            trend_html = f'<div class="ig-card-trend ig-trend-{trend_dir}">{trend}</div>'
            
        html = f"""
        <div class="ig-card">
          <div class="ig-card-title">{icon} {label}</div>
          <div class="ig-card-value">{value}</div>
          {trend_html}
        </div>
        """
        col.markdown(html, unsafe_allow_html=True)

def render_feature_bars(explanation: List[Dict[str, Any]]) -> None:
    """Render SHAP feature importances as horizontal bars."""
    if not explanation:
        st.write("No features to display.")
        return
        
    st.markdown('<div style="margin-top:1rem; display:flex; flex-direction:column; gap:0.5rem;">', unsafe_allow_html=True)
    for feat in explanation:
        name = feat.get("feature", "Unknown")
        val = feat.get("value", 0.0)
        # Handle formatting for floats vs strings
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
            
        contrib = feat.get("contribution", 0.0)
        # Normalize for bar width
        w = min(100, max(5, abs(contrib) * 100))
        color = "var(--danger)" if contrib > 0 else "var(--success)"
        
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem;">
          <div style="flex:0 0 160px; font-size:0.85rem; color:var(--text-light); text-align:right;" class="ig-text-mono">{name}</div>
          <div style="flex:1;">
            <div style="width:{w}%; height:8px; border-radius:4px; background:{color};"></div>
          </div>
          <div style="flex:0 0 80px; font-size:0.85rem; color:var(--text-main);" class="ig-text-mono">{val_str}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def empty_state(icon: str, title: str, subtitle: str) -> None:
    """Render a professional empty state."""
    st.markdown(f'''
    <div class="ig-empty-state">
      <div class="ig-empty-icon">{icon}</div>
      <div class="ig-empty-title">{title}</div>
      <div style="font-size: 0.9rem;">{subtitle}</div>
    </div>
    ''', unsafe_allow_html=True)
