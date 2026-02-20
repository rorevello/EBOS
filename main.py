import gradio as gr
import numpy as np
import pandas as pd
import os
from src import config, data_loader, embeddings, search

import plotly.express as px
from sklearn.decomposition import PCA
import tempfile

import re
from collections import Counter

# Global state to mimic Streamlit's session state
# In a real app with multiple users, this would need to be handled differently (e.g., passing state or using a database)
# Gradio handles state per-user session automatically if we pass it around, but for caching large models/data we use global variables or lru_cache.

STOPWORDS = set([
    'the', 'and', 'of', 'in', 'a', 'to', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'as', 'it', 'are', 'an', 'be', 'from', 'or', 'at', 'which', 'was', 'were', 'has', 'have', 'had', 'been', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
    'ontology', 'ontologies', 'data', 'information', 'system', 'based', 'using', 'model', 'analysis', 'research', 'development', 'application', 'project', 'used', 'defined', 'standard', 'description', 'term', 'concept', 'knowledge', 'domain', 'web', 'semantic', 'owl', 'rdf',
    'le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'en', 'pour', 'que', 'qui', 'dans', 'sur', 'par', 'ce', 'cette', 'ces', 'est', 'sont', 'du', 'au', 'aux', # French stopwords
])

# Load Data (Global Cache)
print("Loading Ontologies...")
OBO_ONTOLOGIES, OBO_DOMAINS = data_loader.get_obo_ontologies()
AVAILABLE_REPOS = data_loader.get_available_repositories()
ALL_ONTOLOGIES = OBO_ONTOLOGIES.copy()
ALL_DOMAINS = OBO_DOMAINS.copy()

# Load Model
print("Loading Model...")
TOKENIZER, MODEL = embeddings.load_model()
CACHED_EMBEDDINGS = {} 
CACHED_IDS = {}
INDEX = None

def generate_analytics_charts(current_ontologies):
    """Generate Universal Analytics: Domain Distribution and Keyword Cloud."""
    if not current_ontologies:
        return None, None

    # 1. Domain Distribution
    all_domains = []
    for data in current_ontologies.values():
        doms = data.get('domain', ['Unknown'])
        if isinstance(doms, list):
            all_domains.extend(doms)
        else:
            all_domains.append(str(doms))
            
    if all_domains:
        domain_counts = Counter(all_domains)
        df_domains = pd.DataFrame(domain_counts.most_common(15), columns=['Domain', 'Count'])
        fig_domains = px.bar(
            df_domains, x='Count', y='Domain', orientation='h', 
            title="Top 15 Domains", 
            template="plotly_white",
            color='Count', color_continuous_scale='Viridis'
        )
        fig_domains.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig_domains = None

    # 2. Keyword Frequency
    text_blob = ""
    for data in current_ontologies.values():
        text_blob += f" {data.get('title', '')} {data.get('description', '')}"
    
    # Simple Tokenization
    words = re.findall(r'\b\w+\b', text_blob.lower())
    filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]
    
    if filtered_words:
        word_counts = Counter(filtered_words)
        df_keywords = pd.DataFrame(word_counts.most_common(20), columns=['Keyword', 'Frequency'])
        fig_keywords = px.bar(
            df_keywords, x='Frequency', y='Keyword', orientation='h',
            title="Top 20 Keywords",
            template="plotly_white",
             color='Frequency', color_continuous_scale='Blues'
        )
        fig_keywords.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig_keywords = None
        
    return fig_domains, fig_keywords

def load_data(selected_repos):
    """Load selected repositories and rebuild index if needed."""
    global ALL_ONTOLOGIES, ALL_DOMAINS, CACHED_EMBEDDINGS, CACHED_IDS, INDEX
    
    current_ontologies = {}
    current_domains = set()
    current_embeds = []
    current_ids = []
    
    for repo in selected_repos:
        if repo == "OBO Foundry":
            onts, doms = OBO_ONTOLOGIES, OBO_DOMAINS
        else:
            onts, doms = data_loader.get_local_repo(repo)
        
        current_ontologies.update(onts)
        current_domains.update(doms)
        
        # Load Embeddings
        embeds, ids = embeddings.load_embeddings(repo)
        
        if embeds is None:
            # Generate if missing
            print(f"Generating embeddings for {repo}...")
            repo_ids = []
            repo_texts = []
            
            for oid, data in onts.items():
                text = f"{data.get('title', '')} {data.get('description', '')}"
                repo_texts.append(text)
                repo_ids.append(oid)
            
            if repo_texts:
                embeds = MODEL.encode(repo_texts, normalize_embeddings=True)
                ids = repo_ids
                # Save
                embeddings.save_embeddings(repo, embeds, ids)
                print(f"Saved embeddings for {repo}")
            else:
                embeds = np.array([])
                ids = []

        if embeds is not None and len(embeds) > 0:
            current_embeds.append(embeds)
            current_ids.extend(ids)

    if current_embeds:
        full_embeds = np.concatenate(current_embeds)
        INDEX = search.create_index(full_embeds)
        CACHED_IDS = current_ids # Simplified list
        CACHED_EMBEDDINGS = full_embeds # Keep reference
    else:
        INDEX = None
        CACHED_IDS = []

    ALL_ONTOLOGIES = current_ontologies
    ALL_DOMAINS = sorted(list(current_domains))
    
    # Generate Global Analytics
    print("Generating Analytics...")
    fig_map = generate_semantic_map()
    fig_domains, fig_keywords = generate_analytics_charts(current_ontologies)
    
    status_text = f"Loaded {len(current_ontologies)} ontologies from {len(selected_repos)} repositories."
    
    return gr.update(choices=["Select All"] + ALL_DOMAINS, value=["Select All"]), status_text, fig_map, fig_domains, fig_keywords

def generate_semantic_map(query_embedding=None):
    """Generate a 2D semantic map using PCA."""
    if INDEX is None or len(CACHED_IDS) < 3:
        return None
    
    # Use PCA to reduce to 2D
    # For performance, if we have too many points, maybe sample? But PCA is fast enough for <10k
    if len(CACHED_EMBEDDINGS) < 2:
        return None # Not enough points for PCA
        
    n_components = min(2, len(CACHED_EMBEDDINGS))
    pca = PCA(n_components=2 if len(CACHED_EMBEDDINGS) >= 2 else 1) # Force 2D usually, but handle edge case
    
    try:
        components = pca.fit_transform(CACHED_EMBEDDINGS)
    except Exception as e:
        print(f"PCA Error: {e}")
        return None

    if components.shape[1] < 2:
        # If we only got 1 component (e.g. 1 sample), duplicate for y
        components = np.column_stack((components, np.zeros(components.shape[0])))
    
    df_plot = pd.DataFrame(components, columns=['x', 'y'])
    df_plot['id'] = CACHED_IDS
    df_plot['title'] = [ALL_ONTOLOGIES.get(oid, {}).get('title', oid) for oid in CACHED_IDS]
    # Handle domain for Map (take first if list)
    domains_list = []
    for oid in CACHED_IDS:
        d = ALL_ONTOLOGIES.get(oid, {}).get('domain', 'Unknown')
        if isinstance(d, list):
            domains_list.append(d[0] if d else "Unknown")
        else:
            domains_list.append(str(d))
    df_plot['domain'] = domains_list
    
    # If there's a query, highlight the query position (projected)
    if query_embedding is not None:
        query_proj = pca.transform(query_embedding.reshape(1, -1))
        # Add query point
        df_query = pd.DataFrame([[query_proj[0][0], query_proj[0][1], "QUERY", "YOUR SEARCH", "Search Query"]], 
                                columns=['x', 'y', 'id', 'title', 'domain'])
        df_plot = pd.concat([df_plot, df_query], ignore_index=True)
        
    fig = px.scatter(
        df_plot, x='x', y='y', 
        hover_data=['title', 'domain'], 
        color='domain',
        title="Semantic Map of Ontologies (PCA Projection)",
        template="plotly_white",
        symbol_sequence=['circle', 'x'] if query_embedding is not None else ['circle'] 
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    if query_embedding is not None:
        # Highlight Query point specifically if needed, logic above handles it via "Search Query" domain/color usually
        pass
        
    return fig

def perform_search(query, selected_domains, num_results):
    if not query:
        return "<div style='text-align:center'>Enter a query...</div>", None, None, None
    
    if INDEX is None:
        return "Index not loaded.", None, None, None


    # Filter domains
    if "Select All" in selected_domains:
        target_domains = ALL_DOMAINS
    else:
        target_domains = selected_domains
        
    query_embedding = embeddings.get_embedding(query, TOKENIZER, MODEL)
    
    # Search
    results = search.search_ontologies(
        query_embedding, 
        INDEX, 
        CACHED_IDS, 
        num_results,
        ontologies_data=ALL_ONTOLOGIES,
        query_text=query
    )
    
    # 1. HTML Cards
    formatted_results = []
    # 2. Data for DataFrame
    table_data = []
    
    for oid, score in results:
        data = ALL_ONTOLOGIES.get(oid)
        if not data: continue
        
        # Domain Filter
        ont_domains = data.get('domain', ['Unknown'])
        if isinstance(ont_domains, list):
            domain_match = any(d in target_domains for d in ont_domains)
            display_domain = ", ".join(ont_domains)
        else:
            domain_match = ont_domains in target_domains
            display_domain = str(ont_domains)
            
        if not domain_match:
            continue
            
        relevance = max(0, 100 - (score * 50))
        
        # HTML
        card_html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <h3 style="margin: 0;">{data.get('title', 'No Title')}</h3>
                <span style="font-weight: bold; color: { 'green' if relevance > 70 else 'orange' };">{relevance:.0f}% Match</span>
            </div>
            <p style="font-size: 0.9em; color: #666;">ID: {oid} | Domain: {display_domain}</p>
            <p>{data.get('description', '')}</p>
            <div style="margin-top: 10px;">
                <a href="{data.get('url_doc', '#')}" target="_blank" style="margin-right: 15px; text-decoration: none; color: #007bff;">📄 Documentation</a>
                <a href="{data.get('url_download', '#')}" target="_blank" style="text-decoration: none; color: #007bff;">📥 Download</a>
            </div>
        </div>
        """
        formatted_results.append(card_html)
        
        # Table
        table_data.append([
            oid, 
            data.get('title', ''), 
            f"{relevance:.1f}%", 
            display_domain, 
            data.get('status', data.get('activity_status', 'Unknown')),
            data.get('license', 'Unknown'),
            data.get('products', 0),
            data.get('description', '')[:100] + "..."
        ])
        
    if not formatted_results:
        html_out = "<div style='padding: 20px; text-align: center;'>No suitable results found.</div>"
        df_out = pd.DataFrame(columns=["ID", "Title", "Score", "Domain", "Status", "License", "Products", "Description"])
        file_out = None
    else:
        html_out = "\n".join(formatted_results)
        df_out = pd.DataFrame(table_data, columns=["ID", "Title", "Score", "Domain", "Status", "License", "Products", "Description"])
        
        # Generate CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            df_out.to_csv(tmp.name, index=False)
            file_out = tmp.name

    # 3. Generate Semantic Map (with query)
    plot_out = generate_semantic_map(query_embedding)

    return html_out, df_out, plot_out, file_out


# UI Layout
with gr.Blocks(title="EBOS") as demo:
    gr.Markdown("# 🧬 EBOS: Embedding-Based Ontology Search")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            repo_selector = gr.CheckboxGroup(
                choices=["OBO Foundry"] + AVAILABLE_REPOS,
                value=["OBO Foundry"],
                label="Repositories"
            )
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            domain_selector = gr.Dropdown(
                choices=["Select All"] + sorted(list(ALL_DOMAINS)),
                value=["Select All"],
                multiselect=True,
                label="Filter by Domain"
            )
            
            num_results_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Results")
            

        with gr.Column(scale=3):
            with gr.Row():
                query_input = gr.Textbox(label="Search Query", placeholder="Enter your search term...", scale=4)
                search_btn = gr.Button("Search", variant="primary", scale=1)
            
            with gr.Tabs():
                with gr.TabItem("Cards"):
                    results_html = gr.HTML(label="Results")
                with gr.TabItem("Table"):
                    results_table = gr.DataFrame(label="Results Table", interactive=False)
                with gr.TabItem("Semantic Map"):
                    results_plot = gr.Plot(label="Semantic Map")
                with gr.TabItem("Analytics"):
                    gr.Markdown("### Universal Analytics")
                    with gr.Row():
                        plot_domains = gr.Plot(label="Domain Distribution")
                        plot_keywords = gr.Plot(label="Top Keywords")
            
            with gr.Row():
                export_file = gr.File(label="Download Results (CSV)")

    # Interactions
    repo_selector.change(load_data, inputs=[repo_selector], outputs=[domain_selector, status_msg, results_plot, plot_domains, plot_keywords])
    
    search_inputs = [query_input, domain_selector, num_results_slider]
    search_outputs = [results_html, results_table, results_plot, export_file] 
    
    search_btn.click(perform_search, inputs=search_inputs, outputs=search_outputs)
    query_input.submit(perform_search, inputs=search_inputs, outputs=search_outputs)

    # Initial Load
    demo.load(load_data, inputs=[repo_selector], outputs=[domain_selector, status_msg, results_plot, plot_domains, plot_keywords])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
