import streamlit as st
import pandas as pd
import altair as alt
import re

def highlight_text(text, query):
    """Highlight query terms in text using HTML/Markdown."""
    if not query or not text:
        return text
    
    # Escape query for regex and be case insensitive
    escaped_query = re.escape(query.strip())
    pattern = re.compile(f"({escaped_query})", re.IGNORECASE)
    
    # Replace with highlighted version (using yellow background)
    return pattern.sub(r':orange[\g<1>]', text)

def render_ontology_card(ontology, score, status_check_func=None, query_text=""):
    """
    Render a card for a single ontology result.
    
    Args:
        ontology (dict): Ontology data.
        score (float): Relevance score (lower is better for L2, but we can invert for display).
        status_check_func (callable, optional): Function to check URL status.
        query_text (str, optional): Search query for highlighting.
    """
    # Calculate relevance percentage (mock calculation for L2 distance)
    # Assuming distance 0 is 100% and >1.5 is 0% (heuristic)
    relevance = max(0, 100 - (score * 50)) 
    
    with st.container(border=True):
        # Header with Title and Score
        c_head1, c_head2 = st.columns([4, 1])
        with c_head1:
             st.subheader(f"{ontology.get('title', 'No Title')}")
        with c_head2:
             st.metric("Relevance", f"{relevance:.0f}%")

        # Metadata Row
        c_meta1, c_meta2, c_meta3 = st.columns([1, 1, 2])
        with c_meta1:
            st.caption(f"**ID**: `{ontology.get('id', 'N/A')}`")
        with c_meta2:
            st.caption(f"**Repo**: {ontology.get('repository', 'Unknown')}")
        with c_meta3:
            dom = ontology.get('domain', 'Unknown')
            if isinstance(dom, list):
                dom = ", ".join(dom)
            st.caption(f"**Domain**: {dom}")

        # Description with Highlighting
        desc = ontology.get('description', 'No description available.')
        if query_text:
            # We can't use st.write for colored text inside a block easily without markdown
            # But highlight_text returns markdown-compatible color syntax for Streamlit
            st.markdown(highlight_text(desc, query_text))
        else:
            st.write(desc)
            
        st.divider()
        
        # Actions
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            if ontology.get('url_doc'):
                st.link_button("📄 Documentation", ontology['url_doc'], use_container_width=True)
        with btn_col2:
            if ontology.get('url_download'):
                st.link_button("📥 Download", ontology['url_download'], use_container_width=True)
        with btn_col3:
            # Copy ID button (using st.code for easy copying)
            st.code(ontology.get('id', ''), language=None)

        # Status Check
        if status_check_func and ontology.get('url_doc'):
            code = status_check_func(ontology['url_doc'])
            color = "green" if code == '200' else "red" if code == 'Error' else "orange"
            st.caption(f"Status: :{color}[{code}]")

def render_domain_chart(ontologies):
    """
    Render a chart showing domain distribution.
    
    Args:
        ontologies (list of dict): List of relevant ontologies.
    """
    domains = []
    for ont in ontologies:
        d = ont.get('domain', 'Unknown')
        if isinstance(d, list):
            domains.extend(d)
        else:
            domains.append(d)
            
    if not domains:
        return

    df = pd.DataFrame(domains, columns=['Domain'])
    domain_counts = df['Domain'].value_counts().reset_index()
    domain_counts.columns = ['Domain', 'Count']
    
    chart = alt.Chart(domain_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Domain", type="nominal"),
        tooltip=['Domain', 'Count']
    ).properties(
        title="Domain Distribution"
    )
    
    st.altair_chart(chart, use_container_width=True)
