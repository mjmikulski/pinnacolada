import numpy as np
import plotly.graph_objects as go


def plot_dupa(trajectories: dict[str, tuple[np.ndarray, np.ndarray]], title: str, filename: str | None = None,
              show_plot: bool = True):
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow']  # You can add more colors if needed
    for i, (n, tr) in enumerate(trajectories.items()):
        fig.add_trace(
            go.Scatter(
                x=tr[0],
                y=tr[1],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)])
            )
        )
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    if filename is not None:
        import kaleido
        print('kaleido version:', kaleido.__version__)
        fig.write_image(filename)
        # fig.write_html('figure.html')

        print(f'Plot saved to {filename}')
    if show_plot:
        fig.show()
