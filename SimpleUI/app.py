from flask import Flask, render_template_string, request, jsonify
import numpy as np
import math
import time
import plotly
import plotly.graph_objects as go
from shortestpath import solve_cqm, solve_nash

app = Flask(__name__)

# Parameters
START_POS = (0.0, 0.0, 0.0)
END_POS = (100.0, 100.0, 0.0)

@app.route('/')
def index():
    # Initial UI
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UAV Shortest Path</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                display: flex;
                margin: 0;
                font-family: Arial, sans-serif;
                background-color: #000;
                color: #fff;
                height: 100vh;
                overflow: hidden;
            }
            .sidebar {
                width: 25%;
                padding: 10px 20px;
                background-color: #333;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                align-items: flex-start;
                height: 100vh;
                box-sizing: border-box;
            }
            .sidebar h2 {
                font-size: 1.5em;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 2px solid #fff;
                width: 100%;
                text-align: left;
            }
            .map-container {
                width: 75%;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                align-items: center;
                padding: 5px;
                height: 100vh;
                box-sizing: border-box;
            }
            .input-group {
                margin-bottom: 20px;
                margin-top: 20px;
                width: 100%;
            }
            .input-group label {
                display: block;
                font-weight: bold;
                color: #ccc;
                margin-bottom: 5px;
            }
            .input-group input, .input-group select {
                width: 100%;
                padding: 8px;
                font-size: 1em;
                background-color: #444;
                color: #ccc;
                border: 1px solid #555;
                box-sizing: border-box;
            }
            .input-group input::placeholder {
                color: #888;
            }
            .submit-btn {
                padding: 10px 15px;
                font-size: 1em;
                color: #fff;
                background-color: #007BFF;
                border: none;
                cursor: pointer;
                margin-top: 15px;
            }
            .submit-btn:hover {
                background-color: #0056b3;
            }
            .tabs {
                margin-bottom: 10px;
            }
            .tabs button {
                padding: 10px 20px;
                font-size: 1em;
                background-color: #444;
                color: #fff;
                border: 1px solid #555;
                cursor: pointer;
                margin-right: 5px;
            }
            .tabs button.active {
                background-color: #007BFF;
                border-color: #0056b3;
            }
            .results-table {
                width: 100%;
                margin-top: 20px;
                color: #ccc;
                border-collapse: collapse;
            }
            .results-table th, .results-table td {
                border: 1px solid #555;
                padding: 10px;
                text-align: left;
            }
            #uav-inputs {
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>UAV Shortest Path</h2>
            <div class="input-group">
                <label for="drones">Number of UAVs</label>
                <input type="number" id="drones" name="drones" min="1" max="10" placeholder="Enter number of UAVs" onchange="createUAVInputs()">
            </div>
            <div id="uav-inputs"></div>
            <div class="input-group">
                <label for="solver_select">Solver to Display</label>
                <select id="solver_select">
                    <option value="cqm">D-Wave (CQM)</option>
                    <option value="nash">Nash Bargaining</option>
                </select>
            </div>
            <button class="submit-btn" onclick="updateMap()">Optimize</button>
        </div>
        <div class="map-container">
            <div class="tabs">
                <button class="tab-btn active" onclick="showGraph()">Graph</button>
                <button class="tab-btn" onclick="showResults()">Results</button>
            </div>
            <div id="graph-container"></div>
            <div id="results-container" style="display:none;">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Path</th>
                            <th>Cost</th>
                            <th>Runtime</th>
                        </tr>
                    </thead>
                    <tbody id="results-body">
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            function showGraph() {
                document.getElementById("graph-container").style.display = "block";
                document.getElementById("results-container").style.display = "none";
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                document.querySelector('.tabs button:nth-child(1)').classList.add('active');
            }

            function showResults() {
                document.getElementById("graph-container").style.display = "none";
                document.getElementById("results-container").style.display = "block";
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                document.querySelector('.tabs button:nth-child(2)').classList.add('active');
            }

            function createUAVInputs() {
                const numDrones = document.getElementById("drones").value;
                const container = document.getElementById("uav-inputs");
                container.innerHTML = "";
                if (numDrones >= 1 && numDrones <= 10) {
                    for (let i = 0; i < numDrones; i++) {
                        const label = String.fromCharCode('A'.charCodeAt(0) + i);
                        container.innerHTML += `
                            <div class="input-group">
                                <label>UAV ${label} (X,Y)</label>
                                <div style="display:flex;gap:5px;">
                                    <input type="number" id="uav${i}_x" name="uav${i}_x" min="0" max="100" placeholder="X">
                                    <input type="number" id="uav${i}_y" name="uav${i}_y" min="0" max="100" placeholder="Y">
                                </div>
                            </div>
                        `;
                    }
                }
            }

            function updateMap() {
                const numDrones = document.getElementById("drones").value;
                const solverChoice = document.getElementById("solver_select").value;

                if (numDrones < 1 || numDrones > 10) {
                    alert("Please enter a number between 1 and 10.");
                    return;
                }

                let uavPositions = [];
                for (let i = 0; i < numDrones; i++) {
                    let xVal = parseFloat(document.getElementById(`uav${i}_x`).value);
                    let yVal = parseFloat(document.getElementById(`uav${i}_y`).value);
                    if (isNaN(xVal) || isNaN(yVal) || xVal < 0 || xVal > 100 || yVal < 0 || yVal > 100) {
                        alert(`Please enter valid coordinates (0-100) for UAV ${String.fromCharCode('A'.charCodeAt(0) + i)}`);
                        return;
                    }
                    uavPositions.push({x: xVal, y: yVal});
                }

                document.getElementById("graph-container").innerHTML = "Loading...";

                fetch(`/optimize?num_drones=${numDrones}&show_solver=${solverChoice}`, {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({uavs: uavPositions})
                })
                .then(response => response.json())
                .then(data => {
                    // Depending on solverChoice, pick edges from either cqm or nash
                    let pathEdges = data.show_solver === 'cqm' ? data.cqm_path_edges : data.nash_path_edges;

                    const nodes = data.positions;
                    const xs = nodes.map(n => n[0]);
                    const ys = nodes.map(n => n[1]);
                    const zs = nodes.map(n => n[2]);

                    const traceNodes = {
                        type: 'scatter3d',
                        mode: 'markers+text',
                        x: xs,
                        y: ys,
                        z: zs,
                        text: data.node_labels,
                        textposition: "top center",
                        marker: {size:5, color:'cyan'},
                        name:'Nodes',
                        showlegend: false
                    };

                    let pathX = [];
                    let pathY = [];
                    let pathZ = [];
                    for (let i=0; i<pathEdges.length; i++) {
                        const e = pathEdges[i];
                        pathX.push(e[0][0], e[1][0], null);
                        pathY.push(e[0][1], e[1][1], null);
                        pathZ.push(e[0][2], e[1][2], null);
                    }

                    const traceEdges = {
                        type:'scatter3d',
                        mode:'lines',
                        x: pathX,
                        y: pathY,
                        z: pathZ,
                        line:{color:'red', width:4},
                        name:'Shortest Path',
                        showlegend: false
                    };

                    const layout = {
                        width: 1000,
                        height: 850,
                        margin: {l:0, r:0, b:0, t:0},
                        showlegend: false,
                        scene: {
                            bgcolor: "#000000",
                            xaxis: { showbackground:false, color:"white" },
                            yaxis: { showbackground:false, color:"white" },
                            zaxis: { showbackground:false, color:"white" }
                        }
                    };

                    Plotly.newPlot('graph-container', [traceNodes, traceEdges], layout);

                    const resultsBody = document.getElementById("results-body");
                    resultsBody.innerHTML = `
                        <tr><td>D-Wave (CQM)</td><td>${data.cqm_path}</td><td>${data.cqm_cost.toFixed(4)}</td><td>${data.cqm_time.toFixed(4)} seconds</td></tr>
                        <tr><td>Nash Bargaining</td><td>${data.nb_path}</td><td>${data.nb_cost.toFixed(4)}</td><td>${data.nb_time.toFixed(4)} seconds</td></tr>
                    `;
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/optimize', methods=['POST','GET'])
def optimize():
    if request.method == 'POST':
        data = request.get_json()
        uav_positions = data['uavs']
    else:
        uav_positions = []
    num_drones = int(request.args.get('num_drones', 1))
    show_solver = request.args.get('show_solver', 'cqm')  # 'cqm' or 'nash'

    node_labels = ['S'] + [chr(ord('A')+i) for i in range(num_drones)] + ['END']
    positions = [(0.0,0.0,0.0)] + [(u['x'], u['y'], 50.0) for u in uav_positions] + [(100.0,100.0,0.0)]

    total_nodes = len(positions)
    adj_matrix = [[float('inf')]*total_nodes for _ in range(total_nodes)]
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                (x1, y1, z1) = positions[i]
                (x2, y2, z2) = positions[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                adj_matrix[i][j] = dist
            else:
                adj_matrix[i][j] = 0.0

    start_idx = 0
    end_idx = total_nodes - 1

    # Force no direct start-to-end edge
    adj_matrix[start_idx][end_idx] = float('inf')

    # Solve CQM (D-Wave)
    cqm_path_labels, cqm_cost, cqm_time, cqm_path_edges = solve_cqm(adj_matrix, start_idx, end_idx, node_labels, positions)

    # Solve Nash Bargaining
    nb_path_labels, nb_cost, nb_time, nb_path_edges = solve_nash(adj_matrix, node_labels, positions)

    response = {
        'cqm_path': cqm_path_labels,
        'cqm_cost': cqm_cost,
        'cqm_time': cqm_time,
        'nb_path': nb_path_labels,
        'nb_cost': nb_cost,
        'nb_time': nb_time,
        'positions': positions,
        'cqm_path_edges': cqm_path_edges,
        'nash_path_edges': nb_path_edges,
        'node_labels': node_labels,
        'show_solver': show_solver
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
