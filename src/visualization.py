import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_performance_chart(history):
    """
    Creates a 2x2 Plotly figure showing metrics history.
    Args:
        history (dict): Dictionary with keys 'acc', 'prec', 'rec', 'f1', 'best_f1'.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy", "Precision", "Recall", "F1 Score"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    fig.update_layout(
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="#333333", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    
    # Common Axis Styling
    for axis in ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4', 'yaxis', 'yaxis2', 'yaxis3', 'yaxis4']:
        fig.update_layout({
            axis: dict(showgrid=True, gridwidth=1, gridcolor='#E5E5EA', zeroline=False, showline=True, linecolor='#333')
        })
    
    # fig.update_yaxes(range=[0, 1.05])  <-- Removed forced range to allow auto-scaling
    
    # If history is empty, return skeleton
    if not history or not history.get('f1'):
        return fig

    x_vals = list(range(1, len(history['f1']) + 1))
    
    metrics_config = [
        (history.get('acc', []), 1, 1, '#00CC96', 'Accuracy'),
        (history.get('prec', []), 1, 2, '#EF553B', 'Precision'),
        (history.get('rec', []), 2, 1, '#AB63FA', 'Recall'),
        (history.get('f1', []), 2, 2, '#FFA15A', 'F1 Score')
    ]
    
    for data_list, r, c, color, name in metrics_config:
        if not data_list: continue
        
        # Trial Line
        fig.add_trace(
                go.Scatter(x=x_vals, y=data_list, mode='lines+markers', name=name, 
                        line=dict(color=color, width=2), marker=dict(size=6)),
                row=r, col=c
        )
        # Update Subtitle
        fig.layout.annotations[(r-1)*2 + (c-1)].text = f"{name}: {data_list[-1]:.4f}"

    # Add "Best So Far" Trace to F1 Chart
    if history.get('best_f1'):
        fig.add_trace(
            go.Scatter(x=x_vals, y=history['best_f1'], mode='lines', name='Best F1',
                    line=dict(color='#2E86C1', width=3, dash='dash')),
            row=2, col=2
        )
        
    return fig
