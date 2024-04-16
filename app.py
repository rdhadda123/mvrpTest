# Copyright 2024 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Union

import dash
import diskcache
import folium
from dash import DiskcacheManager, callback_context, ctx, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import cssutils

from dash_html import create_table, set_html
from map import (generate_mapping_information, plot_solution_routes_on_map,
                 show_locations_on_initial_map)
from solver.solver import RoutingProblemParameters, SamplerType, Solver, VehicleType

from app_configs import DEBUG

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

from app_configs import APP_TITLE, THEME_COLOR

if TYPE_CHECKING:
    from dash import html

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    prevent_initial_callbacks="initial_duplicate",
    background_callback_manager=background_callback_manager,
)
app.title = APP_TITLE

server = app.server
app.config.suppress_callback_exceptions = True

BASE_PATH = Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("input").resolve()

# Generates css file and variable using THEME_COLOR setting
css = f'''/* Generated theme settings css file, see app.py */
:root {{
    --theme: {THEME_COLOR};
}}
'''
sheet = cssutils.parseString(css)
cssTextDecoded = sheet.cssText.decode('ascii')
with open("assets/theme.css", 'w') as f:
    f.write(cssTextDecoded)


@app.callback(
    Output({'type': 'to-collapse-class', 'index': MATCH}, "className"),
    inputs=[
        Input({'type': 'collapse-trigger', 'index': MATCH}, "n_clicks"),
        State({'type': 'to-collapse-class', 'index': MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> str:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not visible, empty string if visible

    Returns:
        str: The new class name of the thing to collapse.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes)
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed"


def generate_inital_map(num_clients: int) -> folium.Map:
    """Generates the initial map.

    Args:
        num_clients (int): Number of force locations.

    Returns:
        folium.Map: Initial map shown on the map tab.
    """
    map_network, depot_id, force_locations, map_bounds = generate_mapping_information(num_clients)
    initial_map = show_locations_on_initial_map(map_network, depot_id, force_locations, map_bounds)
    return initial_map


@app.callback(
    Output("solution-map", "srcDoc", allow_duplicate=True),
    inputs=[
        Input("num-clients-select", "value"),
        Input("run-button", "n_clicks"),
    ],
)
def render_initial_map(num_clients: int, _) -> str:
    """Generates and saves and HTML version of the initial map.

    Note that 'run-button' is required as an Input to reload the map each time
    a run is started. This resets the solution map to the initial map but does
    NOT regenerate the initial map unless 'num-clients-select' is changed.

    Args:
        num_clients: Number of force locations.

    Returns:
        str: Initial map shown on the map tab as HTML.
    """
    map_path = Path("initial_map.html")

    # only regenerate map if num_clients is changed (i.e., if run buttons is NOT clicked)
    if ctx.triggered_id != "run-button" or not map_path.exists():
        initial_map = generate_inital_map(num_clients)
        initial_map.save(map_path)

    return open(map_path, "r").read()


@app.callback(
    Output("solution-cost-table", "children"),
    Output("solution-cost-table-classical", "children"),
    inputs=[
        Input("run-in-progress", "data"),
        State("stored-results", "data"),
        State("reset-results", "data"),
        State("sampler-type", "data"),
    ],
    prevent_initial_call=True,
)
def update_tables(
    run_in_progress, stored_results, reset_results, sampler_type
) -> tuple[list, list]:
    """Update the results tables each time a run is made.

    Args:
        run_in_progress: Whether or not the ``run_optimization`` callback is running.
        stored_results: The results tab from the latest run.
        reset_results: Whether or not to reset the results tables before applying the new one.
        sampler_type: The sampler type used in the latest run (``"quantum"`` or ``"classical"``)

    Returns:
        tuple: A tuple containing the two results tables.
    """
    empty_or_no_update = [] if reset_results else dash.no_update

    if run_in_progress is True:
        raise PreventUpdate

    if sampler_type == "classical":
        return empty_or_no_update, stored_results

    return stored_results, empty_or_no_update


@app.long_callback(
    # update map and results
    Output("solution-map", "srcDoc", allow_duplicate=True),
    Output("stored-results", "data"),
    # store the solver used, whether or not to reset results tabs and the
    # parameter hash value used to detect parameter changes
    Output("sampler-type", "data"),
    Output("reset-results", "data"),
    Output("parameter-hash", "data"),
    # update table values in top results tab
    Output("problem-size", "children"),
    Output("search-space", "children"),
    Output("performance-improvement-quantum", "children"),
    Output("force-elements", "children"),
    Output("vehicles-deployed", "children"),
    Output("cost-comparison", "data"),
    inputs=[
        Input("run-button", "n_clicks"),
        State("vehicle-type-select", "value"),
        State("sampler-type-select", "value"),
        State("num-vehicles-select", "value"),
        State("solver-time-limit", "value"),
        State("num-clients-select", "value"),
        # input and output result table (to update it dynamically)
        State("solution-cost-table", "children"),
        State("parameter-hash", "data"),
        State("cost-comparison", "data"),
    ],
    running=[
        # show cancel button and hide run button, and disable and animate results tab
        (Output("cancel-button", "className"), "", "display-none"),
        (Output("run-button", "className"), "display-none", ""),
        (Output("results-tab", "disabled"), True, False),
        (Output("results-tab", "label"), "Loading...", "Results"),
        # switch to map tab while running
        (Output("tabs", "value"), "map-tab", "map-tab"),
        # block certain callbacks from running until this is done
        (Output("run-in-progress", "data"), True, False),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    run_click: int,
    vehicle_type: Union[VehicleType, int],
    sampler_type: Union[SamplerType, int],
    num_vehicles: int,
    time_limit: float,
    num_clients: int,
    cost_table: list,
    previous_parameter_hash: str,
    cost_comparison: dict,
) -> tuple[str, list, str, bool, str, int, str, str, int, int, dict]:
    """Run the optimization and update map and results tables.

    This is the main optimization function which is called when the Run optimization button is
    clicked. It used all inputs from the drop-down lists, sliders and text entries and runs the
    optimization, updates the run/cancel buttons, animates (and deactivates) the results tab,
    moves focus to the map tab and updates all relevant HTML entries.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        vehicle_type: Either Trucks (``0`` or ``VehicleType.TRUCKS``) or
            Delivery Drones (``1`` or ``VehicleType.DELIVERY_DRONES``).
        sampler_type: Either Quantum Hybrid (DQM) (``0`` or ``SamplerType.DQM``) or
            Classical (K-Means) (``1`` or ``SamplerType.KMEANS``).
        num_vehicles: The number of vehicles.
        time_limit: The solver time limit.
        num_clients: The number of force locations.
        cost_table: The html 'Solution cost' table. Used to update it dynamically.
        previous_parameter_hash: Previous hash string to detect changed parameters
        cost_comparison: Dictionary with solver keys and run cost values

    Returns:
        A tuple containing all outputs to be used when updating the HTML template (in
        ``dash_html,py``). These are:

            solution-map: Updates the 'srcDoc' entry for the 'solution-map' IFrame in the map tab.
                This is the map (initial and solution map).
            stored-results: Stores the Solution cost table in the results tab.
            sampler-type: The sampler used (``"quantum"`` or ``"classical"``).
            reset-results: Whether or not to reset the results tables before applying the new one.
            parameter-hash: Hash string to detect changed parameters.
            problem-size: Updates the problem-size entry in the Solution stats table.
            search-space: Updates the search-space entry in the Solution stats table.
            wall-clock-time-classical: Updates the wall clock time in the Classical table header.
            performance-improvement-quantum: Updates quatum performance improvement message.
            force-elements: Updates the force-elements entry in the Solution stats table.
            vehicles-deployed: Updates the vehicles-deployed entry in the Solution stats table.
            cost-comparison: Keeps track of the difference between classical and hybrid run costs
    """
    if run_click == 0 or ctx.triggered_id != "run-button":
        return ""
    if isinstance(vehicle_type, int):
        vehicle_type = VehicleType(vehicle_type)

    if isinstance(sampler_type, int):
        sampler_type = SamplerType(sampler_type)

    if ctx.triggered_id == "run-button":
        map_network, depot_id, force_locations, map_bounds = generate_mapping_information(num_clients)
        initial_map = show_locations_on_initial_map(map_network, depot_id, force_locations, map_bounds)

        routing_problem_parameters = RoutingProblemParameters(
            map_network=map_network,
            depot_id=depot_id,
            client_subset=force_locations,
            num_clients=num_clients,
            num_vehicles=num_vehicles,
            vehicle_type=vehicle_type,
            sampler_type=sampler_type,
            time_limit=time_limit,
        )
        routing_problem_solver = Solver(routing_problem_parameters)

        # run problem and generate solution (stored in Solver)
        wall_clock_time = routing_problem_solver.generate()

        solution_map, solution_cost = plot_solution_routes_on_map(
            initial_map,
            routing_problem_parameters,
            routing_problem_solver,
        )

        problem_size = num_vehicles * num_clients
        search_space = f"{num_vehicles**num_clients:.2e}"
        wall_clock_time = f"Wall clock time: {wall_clock_time:.3f}s"

        solution_cost = dict(sorted(solution_cost.items()))
        total_cost = defaultdict(int)
        for _, cost_info_dict in solution_cost.items():
            for key, value in cost_info_dict.items():
                total_cost[key] += value

        cost_table = create_table(
            list(solution_cost.values()), list(total_cost.values())
        )
        solution_map.save("solution_map.html")

        parameter_hash = _get_parameter_hash(**callback_context.states)
        if parameter_hash != previous_parameter_hash:
            reset_results = True
        else:
            reset_results = False

        # Calculates cost improvement between DQM and KMEANS
        cost_comparison_percent = 0
        if reset_results:
            cost_comparison = {str(sampler_type.value): total_cost["optimized_cost"]} # Dict keys must be strings because Dash stores data as JSON
        else:
            cost_comparison[str(sampler_type.value)] = total_cost["optimized_cost"]
            if len(cost_comparison) == 2:
                cost_comparison_percent = (1 - cost_comparison[str(SamplerType.DQM.value)]/cost_comparison[str(SamplerType.KMEANS.value)])*100

        return (
            open("solution_map.html", "r").read(),
            cost_table,
            "classical" if sampler_type is SamplerType.KMEANS else "quantum",
            reset_results,
            str(parameter_hash),
            problem_size,
            search_space,
            "The vehicles travel " + str(round(cost_comparison_percent, 2)) + "% less distance using the quantum hybrid solution." if cost_comparison_percent > 0 else "",
            num_clients,
            num_vehicles,
            cost_comparison
        )

    raise PreventUpdate


def _get_parameter_hash(**states) -> str:
    """Calculate a hash string for parameters which reset the results tables."""
    # list of parameter values that will reset the results tables
    # when changed in the app; must be hashable
    items = [
        "vehicle-type-select.value",
        "num-vehicles-select.value",
        "num-clients-select.value",
        "solver-time-limit.value",
    ]
    try:
        return str(hash(itemgetter(*items)(states)))
    except TypeError as e:
        raise TypeError("unhashable problem parameter value") from e


# import the html code and sets it in the app
# creates the visual layout and app (see `dash_html.py`)
set_html(app)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=DEBUG)
