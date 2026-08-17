#!/usr/bin/env python3
"""Interactive Plotly Dash web interface for the SciForge PINN simulation engine.

Queries material specifications from PostgreSQL, manages parameter sliders,
triggers back-to-back training loops, and renders responsive publication-quality charts.
"""

import logging

import plotly.graph_objects as go
import torch
import torch.nn as nn
from dash import Dash, Input, Output, State, callback, dcc, html

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.carrier_pinn.physics import (
    ConstantFieldPhysics,
    PoissonCoupledPhysics,
)
from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialise Dash App with crisp, clean default typography styling
app = Dash(__name__, title="SciForge Physics Portal")

# Establish data architecture client
db_client = DatabaseClient()


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
                    ],
                ),
                # Interactive Visualisation Axis Box Panel
                html.Div(
                    style={"flex": "2"},
                    children=[
                        dcc.Loading(
                            id="loading-panel",
                            type="circle",
                            children=dcc.Graph(
                                id="simulation-graph", style={"height": "500px"}
                            ),
                        )
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("simulation-graph", "figure"),
    Input("run-button", "n_clicks"),
    State("material-dropdown", "value"),
    State("engine-radio", "value"),
    State("epoch-slider", "value"),
    prevent_initial_call=False,
)
def run_live_simulation_callback(
    n_clicks: int, material_id: str, engine_mode: str, total_epochs: int
) -> go.Figure:
    """Reactive callback that queries the database, triggers the target PINN optimisation loop, and updates the Plotly canvas."""
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
    else:
        model = CarrierPINN(output_dim=2)
        physics_engine = PoissonCoupledPhysics(effective_mass=m_eff, permittivity=eps)
        n_left = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        n_right = torch.tensor([[0.1, 0.5]], dtype=torch.float32)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # Step 3: Run the Optimisation loop steps using the placeholder variable
    logger.info(
        f"Dashboard thread initialising {engine_mode.upper()} engine optimisation loop for {total_epochs} epochs..."
    )
    for _ in range(total_epochs + 1):
        optimiser.zero_grad()
        loss_bc = mse(model(x_left), n_left) + mse(model(x_right), n_right)
        residuals_dict = physics_engine.compute_residuals(x_interior, model)

        loss_physics = 0.0
        for key, res in residuals_dict.items():
            if key == "poisson":
                loss_physics += mse(res * 1.0e-3, torch.zeros_like(res))
            else:
                loss_physics += mse(res, torch.zeros_like(res))

        (loss_bc + loss_physics).backward()
        optimiser.step()

    # Step 4: Extract predictions and generate interactive Plotly Graph Objects
    model.eval()
    with torch.no_grad():
        preds = model(x_interior).cpu().numpy()
        np_x = x_interior.cpu().numpy().flatten()

    fig = go.Figure()

    # Render Primary Data Path: Carrier Density (Using clean raw text notation strings)
    fig.add_trace(
        go.Scatter(
            x=np_x,
            y=preds[:, 0],
            name=r"Simulated Carrier Density ($n/N_0$)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Live Solver Outputs: {formula} ({engine_mode.upper()} Mode)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text=r"Normalised Spatial Coordinate ($x/L$)"), gridcolor="#eee"
        ),
        yaxis=dict(
            title=dict(
                text=r"Normalised Carrier Density ($n/N_0$)", font=dict(color="#1f77b4")
            ),
            gridcolor="#eee",
            tickfont=dict(color="#1f77b4"),
        ),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bordercolor="#ffffff"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    if engine_mode == "coupled":
        fig.add_trace(
            go.Scatter(
                x=np_x,
                y=preds[:, 1],
                name=r"Electrostatic Potentialm $\phi$",
                yaxis="y2",
                line=dict(color="#d62728", width=2.5, dash="dash"),
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=dict(
                    text=r"Electrostatic Potential, $\phi$ (V)",
                    font=dict(color="#d62728"),
                ),
                tickfont=dict(color="#d62728"),
                anchor="x",
                overlaying="y",
                side="right",
            )
        )

    return fig


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8050)
