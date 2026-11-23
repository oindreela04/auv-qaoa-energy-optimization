"""Interactive Plotly map: current field, waypoints, optimal tour."""
import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory import create_quiver


def route_map(mission, tour, C, title="Energy-optimal AUV tour"):
    xy = mission.coords
    pad = 1.0
    gx, gy = np.meshgrid(np.linspace(xy[:, 0].min() - pad, xy[:, 0].max() + pad, 10),
                         np.linspace(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 8))
    u, v = mission.current
    fig = create_quiver(gx, gy, np.full_like(gx, u), np.full_like(gy, v), scale=0.5,
                        arrow_scale=0.3, line=dict(color="#9ecae1", width=1.2),
                        name="Ocean current")

    # Tour legs (forest green) with hover showing the directional cost
    for k in range(len(tour)):
        i, j = tour[k], tour[(k + 1) % len(tour)]
        fig.add_trace(go.Scatter(
            x=[xy[i, 0], xy[j, 0]], y=[xy[i, 1], xy[j, 1]], mode="lines",
            line=dict(color="#228b22", width=4), showlegend=(k == 0), name="Optimal tour",
            hovertemplate=f"{mission.names[i]} → {mission.names[j]}<br>"
                          f"cost {C[i, j]:.3f} (reverse {C[j, i]:.3f})<extra></extra>"))
        fig.add_annotation(x=xy[j, 0], y=xy[j, 1], ax=xy[i, 0], ay=xy[i, 1],
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.3, arrowwidth=2, arrowcolor="#228b22",
                           standoff=8, text="")

    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="Waypoints",
        marker=dict(size=14, color="#08306b"), text=mission.names, textposition=["bottom right", "top left", "bottom right", "top right"][:mission.n] if mission.n == 4 else "top center",
        hovertemplate="%{text}<br>(%{x}, %{y})<extra></extra>"))

    fig.update_layout(title=title, template="plotly_white", width=850, height=650,
                      xaxis_title="x (km)", yaxis_title="y (km)",
                      yaxis=dict(scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def _labelpos(n):
    return ["bottom right", "top left", "bottom right", "top right"] if n == 4 else "top center"


def interactive_map(mission, tour, C, compare=None, animate=True, dark=True, frames_per_unit=8):
    """Route map with optional comparison tour and a Play/Pause AUV animation.

    The AUV moves at its effective speed, so it visibly slows down on legs against the current.
    """
    xy = mission.coords
    bg, fg, grid = ("#0b1d33", "#e6f1f5", "#1d3550") if dark else ("#ffffff", "#1b2631", "#e5ecf0")
    pad = 1.0
    x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad
    y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, 11), np.linspace(y0, y1, 9))
    u, v = mission.current
    if np.hypot(u, v) > 1e-9:
        fig = create_quiver(gx, gy, np.full_like(gx, u), np.full_like(gy, v), scale=0.45,
                            arrow_scale=0.3, line=dict(color="#4f9fc7", width=1.1), name="Ocean current")
        fig.data[0].hoverinfo = "skip"
    else:
        fig = go.Figure()

    def add_tour(t, color, name, dash=None, width=4):
        for k in range(len(t)):
            i, j = t[k], t[(k + 1) % len(t)]
            fig.add_trace(go.Scatter(
                x=[xy[i, 0], xy[j, 0]], y=[xy[i, 1], xy[j, 1]], mode="lines",
                line=dict(color=color, width=width, dash=dash), showlegend=(k == 0), name=name,
                legendgroup=name,
                hovertemplate=f"<b>{mission.names[i]} → {mission.names[j]}</b><br>"
                              f"cost {C[i, j]:.3f}<br>reverse {C[j, i]:.3f}<extra>{name}</extra>"))
            fig.add_annotation(x=xy[j, 0], y=xy[j, 1], ax=xy[i, 0], ay=xy[i, 1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2,
                               arrowwidth=2, arrowcolor=color, standoff=9, text="")

    if compare is not None and tuple(compare) != tuple(tour):
        add_tour(compare, "#f4a259", "Compared tour", dash="dash", width=3)
    add_tour(tour, "#2ecc71", "Optimal tour")

    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", name="Waypoints", text=mission.names,
        textposition=_labelpos(mission.n), textfont=dict(color=fg, size=13),
        marker=dict(size=16, color="#13a89e", line=dict(color=fg, width=2)),
        hovertemplate="<b>%{text}</b><br>(%{x}, %{y})<extra></extra>"))

    updatemenus = []
    frames = []
    if animate:
        # positions sampled in time: frames per leg proportional to leg travel time
        pts = []
        for k in range(len(tour)):
            i, j = tour[k], tour[(k + 1) % len(tour)]
            steps = max(3, int(round(C[i, j] * frames_per_unit)))
            for s in np.linspace(0, 1, steps, endpoint=False):
                pts.append((xy[i] + s * (xy[j] - xy[i]), f"{mission.names[i]} → {mission.names[j]}"))
        pts.append((xy[tour[0]], "Back at base"))
        fig.add_trace(go.Scatter(x=[pts[0][0][0]], y=[pts[0][0][1]], mode="markers", name="AUV",
                                 marker=dict(size=22, color="#ffd166", symbol="diamond",
                                             line=dict(color="#1b2631", width=2)),
                                 hovertemplate="AUV<extra></extra>"))
        auv_idx = len(fig.data) - 1
        frames = [go.Frame(data=[go.Scatter(x=[p[0]], y=[p[1]])], traces=[auv_idx], name=str(n))
                  for n, (p, lbl) in enumerate(pts)]
        fig.frames = frames
        updatemenus = [dict(type="buttons", direction="left", x=0.0, y=1.09, xanchor="left",
                            bgcolor="#13a89e", font=dict(color="#0b1d33", size=13),
                            bordercolor="#13a89e", pad=dict(r=6, t=4),
                            buttons=[dict(label="▶  Play AUV", method="animate",
                                          args=[None, dict(frame=dict(duration=60, redraw=False),
                                                           transition=dict(duration=0),
                                                           fromcurrent=True, mode="immediate")]),
                                     dict(label="⏸  Pause", method="animate",
                                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                             mode="immediate")])])]

    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=fg, family="Inter, Segoe UI, sans-serif"),
        height=600, margin=dict(l=40, r=20, t=80, b=40), updatemenus=updatemenus,
        annotations=list(fig.layout.annotations) + ([dict(
            text="▶ Play: the AUV slows down against the current", xref="paper", yref="paper",
            x=1.0, y=-0.12, xanchor="right", yanchor="middle", showarrow=False, font=dict(size=12, color="#9fb7c9"))] if animate else []),
        xaxis=dict(title="x (km)", gridcolor=grid, zeroline=False, range=[x0 - 0.3, x1 + 0.3]),
        yaxis=dict(title="y (km)", gridcolor=grid, zeroline=False, scaleanchor="x", scaleratio=1,
                   range=[y0 - 0.3, y1 + 0.3]),
        legend=dict(orientation="h", y=-0.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=fg)),
        hoverlabel=dict(bgcolor="#13304d", font_color="#e6f1f5"))
    return fig


def convergence(histories):
    fig = go.Figure()
    for label, h in histories.items():
        fig.add_trace(go.Scatter(y=h, mode="lines", name=label))
    fig.update_layout(title="QAOA convergence (COBYLA)", template="plotly_white",
                      xaxis_title="Iteration", yaxis_title="⟨H_C⟩", width=850, height=420)
    return fig
