import streamlit as st
from typing import Dict, Any
from src.ui.components import page_header

def render_customer_search(artifacts: Dict[str, Any]) -> None:
    page_header("Customer Search", "Find and analyze customer profiles across all intelligence domains.")
    
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <h2 style="margin-bottom:2rem;">Search Customer Profiles</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Actually, we can just render the Customer 360 page which has the search built in.
    from src.ui.views.customer_360 import render_customer_360
    # Just redirect seamlessly
    st.session_state.current_page = "Customer 360"
    st.rerun()
