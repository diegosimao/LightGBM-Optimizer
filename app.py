import streamlit as st
import pandas as pd

# Custom Modules
from src.data import load_data_from_source, CURATED_DATASETS
from src.visualization import create_performance_chart
from src.optimization import OptimizationManager

# --- Configuration & Styling ---
st.set_page_config(page_title="AutoML LightGBM", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .reportview-container { background: white; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; color: #333; }
    .stButton>button {
        width: 100%; border-radius: 8px;
        background-image: linear-gradient(90deg, #2E86C1 0%, #3498DB 100%);
        border: none; color: white; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("⚡ LightGBM Optimizer")
st.markdown("### Choose a curated dataset or upload your own to start.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("📂 Data Source")
    
    data_mode = st.radio("Select Source:", ["Curated Datasets", "Upload CSV"], index=0)
    
    source = None
    target_col = None
    drop_cols = None
    
    if data_mode == "Curated Datasets":
        dataset_name = st.selectbox("Choose Dataset:", list(CURATED_DATASETS.keys()))
        info = CURATED_DATASETS[dataset_name]
        
        source = info["url"]
        target_col = info["target"]
        drop_cols = info.get("drop_cols", [])
        
        st.info(f"Target: **{target_col}**")
        st.caption(f"Source: {source}")
        
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            source = uploaded_file
            # Preview for Column Selection
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.success(f"Loaded {df_preview.shape[0]} rows")
                all_cols = df_preview.columns.tolist()
                target_col = st.selectbox("Select Target Column", all_cols, index=len(all_cols)-1)
                uploaded_file.seek(0) # Reset
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.info("Upload a file to proceed.")

    st.divider()
    
    st.header("🔍 Search Space")
    iterations = st.slider("Trials", 10, 100, 30)
    
    est_range = st.slider("n_estimators", 50, 500, (100, 300))
    lr_range = st.slider("learning_rate (log)", -3.0, -0.5, (-2.0, -1.0))
    leaves_range = st.slider("num_leaves", 10, 200, (20, 100))
    depth_range = st.slider("max_depth", -1, 30, (3, 15))
    
    # Enable button only if we have a valid source
    run_btn = st.button("🚀 Start Optimization", type="primary", disabled=(source is None))

# --- UI Interface ---
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

if 'history' not in st.session_state or not run_btn:
    chart_placeholder.plotly_chart(create_performance_chart(None), use_container_width=True)

def update_ui(history, current_metrics, best_f1):
    fig = create_performance_chart(history)
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    with metrics_placeholder.container():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{current_metrics['acc']:.4f}")
        k2.metric("Precision", f"{current_metrics['prec']:.4f}")
        k3.metric("Recall", f"{current_metrics['rec']:.4f}")
        k4.metric("F1 Score", f"{current_metrics['f1']:.4f}", delta=f"Best: {best_f1:.4f}")

if run_btn and source:
    with st.spinner("Downloading & Processing Data..."):
        try:
            # Load Data
            data = load_data_from_source(source, target_column=target_col, drop_cols=drop_cols)
            
            # Search Space
            search_space = {
                'n_estimators': est_range,
                'learning_rate': [10**lr_range[0], 10**lr_range[1]],
                'num_leaves': leaves_range,
                'max_depth': depth_range
            }
            
            # Init Manager
            optimizer = OptimizationManager(data, search_space, callback_fn=update_ui)
            
            study = optimizer.run(n_trials=iterations)
            st.success(f"Best F1 Score: {study.best_value:.4f}")
            st.json(study.best_params)
            
        except Exception as e:
            st.error(f"Error: {e}")
