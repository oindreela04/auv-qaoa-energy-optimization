"""Interactive dashboard: streamlit run app.py"""
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auv_qaoa import Mission, cost_matrix, still_water_matrix, tour_cost
from auv_qaoa import solvers
from auv_qaoa.visualize import interactive_map

st.set_page_config(page_title="QAOA · AUV Energy Optimizer", page_icon="🌊", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg, #0b2545 0%, #13315c 55%, #1f6f8b 100%);
       border-radius: 18px; padding: 26px 30px; color: #e6f1f5; margin-bottom: 18px;}
.hero h1 {color: #ffffff; font-size: 2.0rem; margin: 0 0 4px 0;}
.hero .sub {color: #f4a259; font-style: italic; margin-bottom: 14px;}
.chip {display: inline-block; background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
       border-radius: 999px; padding: 4px 12px; margin: 3px 6px 3px 0; font-size: 0.85rem;}
.chip b {color: #7fe3d6;}
.kpi {background: #f2f8f8; border-radius: 14px; padding: 16px 18px; height: 100%;
      box-shadow: 0 1px 4px rgba(11,37,69,0.08);}
.kpi .label {color: #5c6b73; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {color: #0b2545; font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.kpi .note {color: #13a89e; font-size: 0.85rem;}
.kpi.dark {background: #0b2545;} .kpi.dark .value {color: #f4a259;} .kpi.dark .label {color: #9fb7c9;}
.kpi.dark .note {color: #7fe3d6;}
div.stButton > button {border-radius: 10px; font-weight: 600;}
section[data-testid="stSidebar"] {background: #f4f9fa;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌊 QAOA Asymmetric Energy Optimizer for AUVs</h1>
  <div class="sub">Plan the lowest-energy survey loop for an underwater vehicle in an ocean current, with a quantum optimizer</div>
  <span class="chip"><b>Qiskit</b> QAOA</span>
  <span class="chip"><b>QUBO</b> / Ising</span>
  <span class="chip"><b>Asymmetric</b> TSP</span>
  <span class="chip">Ocean-current physics</span>
</div>""", unsafe_allow_html=True)

DEFAULT_WP = pd.DataFrame({
    "name": ["Hub (0)", "Alpha (1)", "Beta (2)", "Gamma (3)"],
    "x": [0.0, 2.0, 5.0, 6.0], "y": [0.0, 3.0, 1.0, 5.0],
    "role": ["Deployment base", "Oil-spill periphery", "Bathymetric fault line", "Thermal vent cluster"]})
PRESETS = {"Default": (0.8, 0.5), "Calm sea": (0.0, 0.0),
           "Westward": (-1.4, 0.0), "South drift": (0.2, -1.0)}

ss = st.session_state
ss.setdefault("u", 0.8); ss.setdefault("v", 0.5); ss.setdefault("speed", 2.0)
ss.setdefault("wp", DEFAULT_WP.copy()); ss.setdefault("result", None); ss.setdefault("wp_ver", 0)

def set_preset(pu, pv):
    ss.u, ss.v = pu, pv


def reset_mission():
    ss.u, ss.v, ss.speed = 0.8, 0.5, 2.0
    ss.wp = DEFAULT_WP.copy(); ss.result = None; ss.wp_ver += 1


# ---------- sidebar controls ----------
with st.sidebar:
    st.header("🧭 Mission controls")
    st.caption("Current presets")
    cols = st.columns(2)
    for k, (name, (pu, pv)) in enumerate(PRESETS.items()):
        cols[k % 2].button(name, width="stretch", on_click=set_preset, args=(pu, pv))
    st.slider("Current u: eastward", -1.8, 1.8, key="u", step=0.1)
    st.slider("Current v: northward", -1.8, 1.8, key="v", step=0.1)
    st.slider("AUV still-water speed", 1.0, 4.0, key="speed", step=0.1)

    st.divider()
    st.subheader("⚛️ Solver")
    method = st.radio("Method", ["Exact (brute force)", "QAOA: reduced encoding", "QAOA: full encoding"],
                      index=1, help="Reduced encoding fixes the base at step 0: (N-1)² qubits instead of N².")
    with st.expander("QAOA settings", expanded=False):
        reps = st.slider("Layers p", 1, 5, 3)
        maxiter = st.slider("COBYLA iterations", 20, 300, 150, step=10)
        shots = st.select_slider("Shots", [512, 1024, 2048, 4096, 8192], value=4096)
        seed = st.number_input("Random seed", 1, 9999, 1)
    run = st.button("🚀 Run optimization", type="primary", width="stretch")
    st.button("↺ Reset mission", width="stretch", on_click=reset_mission)

# ---------- waypoints editor ----------
with st.expander("📍 Edit waypoints (3 to 5 points; the first row is the base)", expanded=False):
    wp = st.data_editor(ss.wp, num_rows="dynamic", width="stretch", key=f"wp_editor_{ss.wp_ver}",
                        column_config={"x": st.column_config.NumberColumn(format="%.1f"),
                                       "y": st.column_config.NumberColumn(format="%.1f")})
wp = wp.dropna(subset=["x", "y"]).reset_index(drop=True)
if not 3 <= len(wp) <= 5:
    st.error("Please keep between 3 and 5 waypoints."); st.stop()
if wp[["x", "y"]].duplicated().any():
    st.error("Two waypoints share the same position."); st.stop()

mission = Mission(names=[str(n) if pd.notna(n) and str(n) else f"P{k}" for k, n in enumerate(wp["name"])],
                  coords=wp[["x", "y"]].to_numpy(float), current=np.array([ss.u, ss.v]),
                  base_speed=float(ss.speed))
try:
    C = cost_matrix(mission)
except ValueError as e:
    st.error(f"⚠️ {e} Lower the current or raise the AUV speed."); st.stop()
S = still_water_matrix(mission)
N = mission.n
ranked = solvers.brute_force(C)
best_cost, best = ranked[0]

# ---------- run solver ----------
qubits = (N - 1) ** 2 if "reduced" in method else N ** 2
if run:
    if method.startswith("QAOA") and qubits > 16:
        st.warning(f"{qubits} qubits is too many for the simulator here. Use the reduced encoding or fewer waypoints.")
    elif method.startswith("Exact"):
        ss.result = dict(solver="Exact (brute force)", tour=best, energy=best_cost, seconds=0.0,
                         qubits=0, history=[], feasible_prob=None, optimal_prob=None)
    else:
        with st.spinner(f"Running QAOA on {qubits} simulated qubits…"):
            r = solvers.qaoa_isolated(C, reduced="reduced" in method, reps=reps, maxiter=maxiter,
                             shots=shots, seed=int(seed), optimum=best)
        r["solver"] = method
        ss.result = r
    if ss.result is not None:
        ss.result["signature"] = (tuple(C.round(6).flatten()))
res = ss.result
stale = res is not None and res.get("signature") != tuple(C.round(6).flatten())

# ---------- KPIs ----------
asym = max(C[i, j] / C[j, i] for i in range(N) for j in range(N) if i != j)
worst = ranked[-1][0]
k1, k2, k3, k4 = st.columns(4)
def kpi(col, label, value, note, dark=False):
    col.markdown(f'<div class="kpi {"dark" if dark else ""}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)
kpi(k1, "Best tour", "→".join(map(str, best)) + f"→{best[0]}", f"energy {best_cost:.3f}", dark=True)
kpi(k2, "Saving vs worst tour", f"{100 * (1 - best_cost / worst):.1f}%", f"worst = {worst:.3f}")
kpi(k3, "Max asymmetry", f"{asym:.2f}×", "costliest leg vs. its reverse")
kpi(k4, "Current speed", f"{np.hypot(ss.u, ss.v):.2f}", f"AUV speed {ss.speed:.1f}")
st.write("")

if res is None:
    st.info("👈 Pick a solver and press **Run optimization**. The map below already shows the brute-force optimum.")
elif stale:
    st.warning("The mission changed since the last run. Press **Run optimization** again.")
else:
    ok = res["tour"] == best
    msg = (f"**{res['solver']}** → tour `{res['tour']}`" + (f", energy {res['energy']:.3f}" if res["tour"] else "")
           + (f" · {res['qubits']} qubits · {res['seconds']:.1f}s" if res["qubits"] else ""))
    (st.success if ok else st.warning if res["tour"] else st.error)(
        ("✅ Optimal! " if ok else "⚠️ Not optimal. " if res["tour"] else "❌ No valid tour found. ") + msg)

# ---------- tabs ----------
t1, t2, t3, t4, t5 = st.tabs(["🗺️ Route map", "📊 All tours", "🔢 Cost matrix", "⚛️ QAOA insight", "ℹ️ About"])

with t1:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Map options**")
        animate = st.toggle("AUV animation", value=True)
        dark = st.toggle("Dark ocean theme", value=True)
        labels = ["None"] + ["-".join(map(str, t)) + f"-{t[0]}  ({c:.2f})" for c, t in ranked[1:]]
        cmp_label = st.selectbox("Compare with tour", labels)
        compare = None if cmp_label == "None" else ranked[labels.index(cmp_label)][1]
        shown = res["tour"] if (res and not stale and res["tour"]) else best
        st.markdown("**Legs of shown tour**")
        legs = pd.DataFrame([{"leg": f"{shown[k]}→{shown[(k + 1) % N]}",
                              "cost": C[shown[k], shown[(k + 1) % N]],
                              "reverse": C[shown[(k + 1) % N], shown[k]]} for k in range(N)])
        st.dataframe(legs.style.format({"cost": "{:.3f}", "reverse": "{:.3f}"}), hide_index=True,
                     width="stretch")
    with c1:
        st.plotly_chart(interactive_map(mission, shown, C, compare=compare, animate=animate, dark=dark),
                        width="stretch")

with t2:
    df = pd.DataFrame([{"Tour": "-".join(map(str, t)) + f"-{t[0]}", "Energy (current)": c,
                        "Still water": tour_cost(t, S), "vs optimum": f"+{100 * (c / best_cost - 1):.1f}%"}
                       for c, t in ranked])
    fig = go.Figure(go.Bar(x=df["Energy (current)"], y=df["Tour"], orientation="h",
                           marker_color=["#13a89e"] + ["#8da9c4"] * (len(df) - 1),
                           text=[f"{v:.2f}" for v in df["Energy (current)"]], textposition="outside"))
    fig.add_trace(go.Scatter(x=df["Still water"], y=df["Tour"], mode="markers", name="Still water",
                             marker=dict(symbol="line-ns-open", size=18, color="#f4a259", line=dict(width=3))))
    fig.update_layout(template="plotly_white", height=max(300, 34 * len(df) + 100), showlegend=False,
                      yaxis=dict(autorange="reversed"), xaxis_title="Energy (time units)",
                      margin=dict(l=10, r=40, t=30, b=40), title="Bars: with current · orange tick: still water")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.style.format({"Energy (current)": "{:.3f}", "Still water": "{:.3f}"}),
                 hide_index=True, width="stretch")

with t3:
    lab = [f"{k}" for k in range(N)]
    Cd = np.where(np.eye(N, dtype=bool), np.nan, C)
    hm = go.Figure(go.Heatmap(z=Cd, x=lab, y=lab, colorscale="Tealrose", text=np.round(Cd, 3),
                              texttemplate="%{text}", hovertemplate="from %{y} to %{x}: %{z:.3f}<extra></extra>"))
    hm.update_layout(height=430, xaxis_title="to node", yaxis_title="from node",
                     yaxis=dict(autorange="reversed"), margin=dict(t=30))
    a, b = st.columns([3, 2])
    a.plotly_chart(hm, width="stretch")
    b.markdown("#### How costs are computed")
    b.latex(r"V_{\text{eff},ij} = V_b + \mathbf{V}_c \cdot \hat{\mathbf{d}}_{ij}")
    b.latex(r"C_{ij} = \frac{\lVert \mathbf{p}_j - \mathbf{p}_i \rVert}{V_{\text{eff},ij}}")
    b.markdown("Reversing a leg flips the sign of the current term, so **C<sub>ij</sub> ≠ C<sub>ji</sub>**. "
               "Read row → column: the darker the cell, the more energy the leg costs.", unsafe_allow_html=True)

with t4:
    if res and res.get("history") and not stale:
        a, b = st.columns(2)
        cv = go.Figure(go.Scatter(y=res["history"], mode="lines", line=dict(color="#13a89e", width=2)))
        cv.update_layout(template="plotly_white", height=360, title="COBYLA convergence of ⟨H_C⟩",
                         xaxis_title="Iteration", yaxis_title="Expected cost")
        a.plotly_chart(cv, width="stretch")
        n_states = 2 ** res["qubits"]
        n_feasible = math.factorial(N - 1) if "reduced" in res["solver"] else math.factorial(N)
        bars = pd.DataFrame({"": ["Valid tours", "Optimal tour"],
                             "QAOA": [res["feasible_prob"] * 100, (res["optimal_prob"] or 0) * 100],
                             "Random guess": [100 * n_feasible / n_states,
                                              100 * (1 if "reduced" in res["solver"] else N) / n_states]})
        pb = go.Figure([go.Bar(name="QAOA", x=bars[""], y=bars["QAOA"], marker_color="#13a89e"),
                        go.Bar(name="Random guess", x=bars[""], y=bars["Random guess"], marker_color="#c9d6df")])
        pb.update_layout(template="plotly_white", height=360, barmode="group", yaxis_title="Probability (%)",
                         title="Is QAOA better than random guessing?")
        b.plotly_chart(pb, width="stretch")
        st.caption(f"{res['qubits']} qubits → {n_states:,} possible bitstrings, of which only {n_feasible} encode valid tours. "
                   "Penalty-based QUBOs waste most of the search space; that is why the reduced encoding helps.")
    else:
        st.info("Run a **QAOA** solver to see convergence and probability analysis here.")

with t5:
    st.markdown("""
**Problem.** An AUV must visit every survey point and return to base on one battery. Currents make each leg's
cost depend on direction, so the task is an *Asymmetric Traveling Salesperson Problem*.

**Pipeline.** Ocean physics → asymmetric cost matrix → QUBO (DOcplex + Qiskit Optimization) → Ising Hamiltonian
→ QAOA (COBYLA) → decoded tour, checked against brute force.

**Honest note.** With 3 to 5 waypoints a classical computer solves this instantly. The project shows *how* a
real-world routing problem is mapped to a quantum optimizer. It does not show quantum advantage.
""")
