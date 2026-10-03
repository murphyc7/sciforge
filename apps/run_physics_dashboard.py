#!/usr/bin/env python3
"""Interactive Plotly Dash web interface for the SciForge PINN simulation engine.

Queries material specifications from PostgreSQL, manages parameter sliders, dispatches
PINN optimisation loops onto background threads, and renders responsive publication-quality
charts of both live training progress and the final solver output.
"""

import logging
import os
import queue
import threading
import uuid
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from dash import Dash, Input, Output, State, callback, dcc, exceptions, html, no_update

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    BipolarCoupledPhysics,
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
from sciforge.matkit.carrier_pinn.training_worker import (
    PINNTrainingWorker,
    TrainingMetrics,
)
from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialise Dash App
app = Dash(
    __name__,
    title="SciForge Physics Portal",
)

# Explicitly enforce your verified development credentials if the
# environment terminal has not broadcasted the DATABASE_URL variable.
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = (
        "postgresql://postgres_user:secure_pass@localhost:5432/sciforge_db"
    )

# Establish data architecture client
db_client = DatabaseClient()

# Poll cadence for draining each background worker's metrics queue. Short enough to feel
# live, long enough not to spam the Flask dev server with requests.
METRICS_POLL_INTERVAL_MS = 400


@dataclass
class _TrainingSession:
    """Bookkeeping record for one in-flight or just-completed background training run.

    Attributes:
        worker (PINNTrainingWorker): The decoupled training harness owning the metrics
            queue that this session's browser tab polls.
        thread (threading.Thread): The daemon thread executing `worker.run`.
        stop_event (threading.Event): Cooperative cancellation flag for this run.
        engine_mode (str): Selected solver profile flag ('constant'/'coupled'/'bipolar').
        formula (str): Chemical formula of the material this run was trained against.
    """

    worker: PINNTrainingWorker
    thread: threading.Thread
    stop_event: threading.Event
    engine_mode: str
    formula: str


# Concurrent training sessions, keyed by a UUID handed to the browser via dcc.Store. Each
# browser tab drives its own session, so multiple users (or multiple runs in one tab) can
# train concurrently without their metrics queues colliding. Guarded by _sessions_lock
# since the Flask dev server can field requests from more than one thread at once.
_sessions: dict[str, _TrainingSession] = {}
_sessions_lock = threading.Lock()


def get_db_material_options() -> list[dict[str, str]]:
    """Queries live PostgreSQL table records to construct dropdown UI selectors.

    Returns:
        list[dict[str, str]]: Dropdown label/value dictionaries.
    """
    try:
        with db_client.get_session() as session:
            records = session.query(MaterialModel).all()
            return [
                {"label": f"{r.formula} ({r.material_id})", "value": r.material_id}
                for r in records
            ]
    except Exception as e:
        logger.error(
            f"Failed to query dropdown material choices from SQL database: {e}"
        )
        return [{"label": "GaAs (Fallback-Mock)", "value": "mp-100"}]


def _build_progress_figure(history: list[dict[str, float]]) -> go.Figure:
    """Renders the live loss-convergence chart from accumulated metrics snapshots.

    Args:
        history (list[dict[str, float]]): Ordered metrics rows accumulated client-side
            across polling ticks, each with 'epoch', 'total_loss', 'bc_loss' and
            'physics_loss' keys.

    Returns:
        go.Figure: A log-scale convergence plot tracking all three loss components.
    """
    fig = go.Figure()
    if history:
        epochs = [row["epoch"] for row in history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[row["total_loss"] for row in history],
                name="Total Loss",
                line=dict(color="#1f77b4", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[row["bc_loss"] for row in history],
                name="Boundary Mismatch",
                line=dict(color="#ff7f0e", width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[row["physics_loss"] for row in history],
                name="Physics Residual",
                line=dict(color="#2ca02c", width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=dict(text="Live Optimisation Convergence", font=dict(size=14)),
        xaxis=dict(title=dict(text="Epoch"), gridcolor="#eee"),
        yaxis=dict(title=dict(text="Loss"), type="log", gridcolor="#eee"),
        template="plotly_white",
        legend=dict(x=0.98, y=0.98, xanchor="right", bordercolor="#ffffff"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=220,
    )
    return fig


def _build_result_figure(
    predictions: np.ndarray, engine_mode: str, formula: str
) -> go.Figure:
    """Renders the final solver output: carrier density (and potential, if solved).

    Args:
        predictions (np.ndarray): Evaluation-mode model outputs over the interior grid,
            shape [N, output_dim].
        engine_mode (str): Selected solver profile flag ('constant'/'coupled'/'bipolar').
        formula (str): Chemical formula of the material this run was trained against.

    Returns:
        go.Figure: The publication-style dual-axis transport profile figure.
    """
    np_x = np.linspace(0.0, 1.0, predictions.shape[0])
    fig = go.Figure()

    # Render Primary Data Path: Carrier Density (Using clean raw text notation strings)
    fig.add_trace(
        go.Scatter(
            x=np_x,
            y=predictions[:, 0],
            name=r"Simulated Carrier Density (n/N₀)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    # Dynamic bipolar trace allocation path
    if engine_mode == "bipolar":
        # Extract hole density from channel 1 and map it to the primary Y-axis
        fig.add_trace(
            go.Scatter(
                x=np_x,
                y=predictions[:, 1],
                name="Simulated Holes ($p/N_0$)",
                line=dict(color="#9467bd", width=2.5, dash="dashdot"),
            )
        )
        # Potential maps to channel 2 in bipolar mode
        phi_data = predictions[:, 2]
    else:
        # Potential maps to channel 1 in traditional unipolar coupled mode
        phi_data = predictions[:, 1] if engine_mode == "coupled" else None

    if phi_data is not None:
        fig.add_trace(
            go.Scatter(
                x=np_x,
                y=phi_data,
                name=r"Electrostatic Potential, ϕ",
                yaxis="y2",
                line=dict(color="#d62728", width=2.5, dash="dash"),
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=dict(
                    text=r"Electrostatic Potential, ϕ (V)",
                    font=dict(color="#d62728"),
                ),
                tickfont=dict(color="#d62728"),
                anchor="x",
                overlaying="y",
                side="right",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Live Solver Outputs: {formula} ({engine_mode.upper()} Mode)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text=r"Normalised Spatial Coordinate (x/L)"), gridcolor="#eee"
        ),
        yaxis=dict(
            title=dict(
                text=r"Normalised Carrier Density (n/N₀)", font=dict(color="#1f77b4")
            ),
            gridcolor="#eee",
            tickfont=dict(color="#1f77b4"),
        ),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bordercolor="#ffffff"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# Define clean, modular layout components
app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "30px",
        "maxWidth": "1200px",
        "margin": "0 auto",
    },
    children=[
        html.H1(
            "SciForge Interactive Physics Dashboard",
            style={
                "color": "#1f77b4",
                "borderBottom": "2px solid #1f77b4",
                "paddingBottom": "10px",
            },
        ),
        html.P(
            "Select persistent material attributes from PostgreSQL to steer live Physics-Informed Neural Network optimisation loops.",
            style={"color": "#555"},
        ),
        html.Div(
            style={"display": "flex", "gap": "30px", "marginTop": "20px"},
            children=[
                # Control Side-Panel Configuration Box
                html.Div(
                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                    },
                    children=[
                        html.Label(
                            "Target Material Source (Live SQL):",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="material-dropdown",
                            options=get_db_material_options(),
                            value="mp-100",
                            clearable=False,
                            style={"marginBottom": "20px"},
                        ),
                        html.Label(
                            "Solver Engine Architecture Configuration:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.RadioItems(
                            id="engine-radio",
                            options=[
                                {"label": " Constant Field (1D)", "value": "constant"},
                                {
                                    "label": " Self-Consistent Coupled Poisson",
                                    "value": "coupled",
                                },
                                {
                                    "label": " Bipolar Coupled Recombination",
                                    "value": "bipolar",
                                },
                            ],
                            value="constant",
                            style={"margin": "10px 0 20px 0"},
                        ),
                        html.Label(
                            "Optimisation Epoch Steps Bounds:",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Slider(
                            id="epoch-slider",
                            min=100,
                            max=1000,
                            step=100,
                            value=300,
                            marks={i: str(i) for i in range(100, 1001, 200)},
                            className="rc-slider",
                        ),
                        html.Button(
                            "Execute Simulation Run",
                            id="run-button",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "backgroundColor": "#2ca02c",
                                "color": "white",
                                "border": "none",
                                "padding": "12px",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginTop": "30px",
                            },
                        ),
                        html.Div(
                            id="training-status",
                            children="Idle — configure a run and click Execute.",
                            style={
                                "marginTop": "14px",
                                "color": "#555",
                                "fontSize": "13px",
                            },
                        ),
                    ],
                ),
                # Interactive Visualisation Axis Box Panel
                html.Div(
                    style={"flex": "2"},
                    children=[
                        dcc.Graph(id="progress-graph", style={"height": "220px"}),
                        dcc.Loading(
                            id="loading-panel",
                            type="circle",
                            children=dcc.Graph(
                                id="simulation-graph",
                                style={"height": "440px"},
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # Decoupled background-thread plumbing: holds this tab's session handle and
        # accumulated metrics history, and drives the real-time poll.
        dcc.Store(id="session-id-store"),
        dcc.Store(id="metrics-history-store", data=[]),
        dcc.Interval(
            id="metrics-interval", interval=METRICS_POLL_INTERVAL_MS, disabled=True
        ),
    ],
)


@callback(
    Output("session-id-store", "data"),
    Output("metrics-history-store", "data"),
    Output("metrics-interval", "disabled"),
    Output("run-button", "disabled"),
    Output("training-status", "children"),
    Input("run-button", "n_clicks"),
    State("material-dropdown", "value"),
    State("engine-radio", "value"),
    State("epoch-slider", "value"),
    prevent_initial_call=True,
)
def start_training_callback(
    n_clicks: int, material_id: str, engine_mode: str, total_epochs: int
) -> tuple[str, list, bool, bool, str]:
    """Builds the model/physics backend and dispatches training onto a background thread.

    Returns immediately once the thread is launched; this callback never blocks on the
    optimisation loop itself, so the Dash server stays responsive for the duration of
    training. Live progress is picked up separately by `poll_training_metrics_callback`.
    """
    # Step 1: Query material parameters from PostgreSQL
    with db_client.get_session() as session:
        material = (
            session.query(MaterialModel).filter_by(material_id=material_id).first()
        )
        m_eff = material.effective_mass_me if material else 0.067
        eps = material.permittivity if material else 12.9
        formula = material.formula if material else "GaAs"

    # Step 2: Set up training simulation grids
    x_interior = torch.linspace(0.0, 1.0, 100, dtype=torch.float32).view(-1, 1)
    x_left = torch.tensor([[0.0]], dtype=torch.float32)
    x_right = torch.tensor([[1.0]], dtype=torch.float32)

    if engine_mode == "constant":
        model = CarrierPINN(output_dim=1)
        physics_engine = ConstantFieldPhysics(effective_mass=m_eff, permittivity=eps)
        n_left = torch.tensor([[1.0]], dtype=torch.float32)
        n_right = torch.tensor([[0.0]], dtype=torch.float32)
    elif engine_mode == "coupled":
        model = CarrierPINN(output_dim=2)
        physics_engine = PoissonCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        n_left = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        n_right = torch.tensor([[0.1, 0.5]], dtype=torch.float32)
    else:
        model = CarrierPINN(output_dim=3)
        physics_engine = BipolarCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        # Sets boundary layers mapping [n, p, phi] for the bipolar system mesh
        n_left = torch.tensor([[1.0, 0.01, 0.0]], dtype=torch.float32)
        n_right = torch.tensor([[0.1, 1.0, 0.8]], dtype=torch.float32)

    # Step 3: Hand the loop off to a background thread, decoupled from this callback
    metrics_queue: queue.Queue[TrainingMetrics] = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    worker = PINNTrainingWorker(
        model=model,
        physics_engine=physics_engine,
        x_interior=x_interior,
        x_left=x_left,
        x_right=x_right,
        n_left=n_left,
        n_right=n_right,
        total_epochs=total_epochs,
        metrics_queue=metrics_queue,
        stop_event=stop_event,
    )
    thread = threading.Thread(target=worker.run, daemon=True)

    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = _TrainingSession(
            worker=worker,
            thread=thread,
            stop_event=stop_event,
            engine_mode=engine_mode,
            formula=formula,
        )

    logger.info(
        f"Dispatching background {engine_mode.upper()} training thread "
        f"(session={session_id}, epochs={total_epochs})."
    )
    thread.start()

    status = f"Training {formula} ({engine_mode.upper()}) — epoch 0 / {total_epochs}"
    return session_id, [], False, True, status


@callback(
    Output("progress-graph", "figure"),
    Output("simulation-graph", "figure"),
    Output("metrics-history-store", "data", allow_duplicate=True),
    Output("metrics-interval", "disabled", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("training-status", "children", allow_duplicate=True),
    Input("metrics-interval", "n_intervals"),
    State("session-id-store", "data"),
    State("metrics-history-store", "data"),
    prevent_initial_call=True,
)
def poll_training_metrics_callback(
    _n_intervals: int, session_id: str | None, history: list[dict[str, float]]
) -> tuple:
    """Drains this session's metrics queue and refreshes both charts in real time.

    Runs on every `dcc.Interval` tick. Each tick, every snapshot currently sitting on the
    worker's queue is pulled off non-blockingly; if the terminal snapshot is among them,
    the final solver-output chart is rendered and the session is retired.
    """
    if session_id is None:
        raise exceptions.PreventUpdate

    with _sessions_lock:
        training_session = _sessions.get(session_id)
    if training_session is None:
        raise exceptions.PreventUpdate

    new_snapshots: list[TrainingMetrics] = []
    while True:
        try:
            new_snapshots.append(training_session.worker.metrics_queue.get_nowait())
        except queue.Empty:
            break

    if not new_snapshots:
        raise exceptions.PreventUpdate  # Nothing queued yet; wait for the next tick.

    history = history + [
        {
            "epoch": m.epoch,
            "total_loss": m.total_loss,
            "bc_loss": m.bc_loss,
            "physics_loss": m.physics_loss,
        }
        for m in new_snapshots
        if m.error is None
    ]
    progress_fig = _build_progress_figure(history)

    final_snapshot = next((m for m in reversed(new_snapshots) if m.is_final), None)
    if final_snapshot is None:
        latest = new_snapshots[-1]
        status = (
            f"Training {training_session.formula} "
            f"({training_session.engine_mode.upper()}) — epoch {latest.epoch}"
        )
        return progress_fig, no_update, history, False, True, status

    # Terminal snapshot received: the background thread has stopped. Retire the session
    # regardless of outcome so its queue and thread reference can be garbage collected.
    with _sessions_lock:
        _sessions.pop(session_id, None)

    if final_snapshot.error is not None:
        status = f"Training failed: {final_snapshot.error}"
        return progress_fig, no_update, history, True, False, status

    result_fig = _build_result_figure(
        predictions=final_snapshot.predictions,
        engine_mode=training_session.engine_mode,
        formula=training_session.formula,
    )
    status = (
        f"Completed: {training_session.formula} "
        f"({training_session.engine_mode.upper()}) — "
        f"{final_snapshot.epoch} epochs in {final_snapshot.elapsed_seconds:.1f}s"
    )
    return progress_fig, result_fig, history, True, False, status


if __name__ == "__main__":
    # threaded=True lets Flask field the metrics-polling requests concurrently with the
    # page's own request cycle while a training thread is running; use_reloader=False
    # avoids Werkzeug's auto-reloader spawning a second process that would not share the
    # in-memory _sessions registry with the one actually serving the browser.
    app.run(debug=True, host="localhost", port=8050, threaded=True, use_reloader=False)
