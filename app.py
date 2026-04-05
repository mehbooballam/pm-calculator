from flask import Flask, render_template, request, jsonify
from collections import defaultdict, deque

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/evm", methods=["POST"])
def calculate_evm():
    """Calculate EVM metrics: CPI, SPI, TCPI, EAC, ETC, VAC, CV, SV."""
    data = request.json
    try:
        bac = float(data["bac"])
        ev = float(data["ev"])
        pv = float(data["pv"])
        ac = float(data["ac"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Provide valid numeric values for BAC, EV, PV, and AC."}), 400

    if bac <= 0:
        return jsonify({"error": "BAC must be greater than zero."}), 400

    # Cost Variance & Schedule Variance
    cv = ev - ac
    sv = ev - pv

    # CPI and SPI
    cpi = ev / ac if ac != 0 else None
    spi = ev / pv if pv != 0 else None

    # EAC variants
    eac_cpi = bac / cpi if cpi and cpi != 0 else None
    eac_ac = ac + (bac - ev)
    eac_combined = (ac + (bac - ev) / (cpi * spi)) if cpi and spi and (cpi * spi) != 0 else None

    # ETC (Estimate to Complete)
    etc = eac_cpi - ac if eac_cpi is not None else None

    # VAC (Variance at Completion)
    vac = bac - eac_cpi if eac_cpi is not None else None

    # TCPI (To-Complete Performance Index)
    tcpi_bac = (bac - ev) / (bac - ac) if (bac - ac) != 0 else None
    tcpi_eac = (bac - ev) / (eac_cpi - ac) if eac_cpi and (eac_cpi - ac) != 0 else None

    # Percent complete
    pct_complete = (ev / bac) * 100
    pct_spent = (ac / bac) * 100

    def r(v):
        return round(v, 4) if v is not None else "N/A"

    return jsonify({
        "cv": r(cv),
        "sv": r(sv),
        "cpi": r(cpi),
        "spi": r(spi),
        "eac_cpi": r(eac_cpi),
        "eac_ac": r(eac_ac),
        "eac_combined": r(eac_combined),
        "etc": r(etc),
        "vac": r(vac),
        "tcpi_bac": r(tcpi_bac),
        "tcpi_eac": r(tcpi_eac),
        "pct_complete": r(pct_complete),
        "pct_spent": r(pct_spent),
    })


@app.route("/api/critical-path", methods=["POST"])
def calculate_critical_path():
    """Calculate critical path using forward/backward pass."""
    data = request.json
    tasks = data.get("tasks", [])

    if not tasks:
        return jsonify({"error": "Provide at least one task."}), 400

    # Build graph
    task_map = {}
    for t in tasks:
        tid = t["id"]
        task_map[tid] = {
            "id": tid,
            "name": t.get("name", tid),
            "duration": float(t["duration"]),
            "predecessors": t.get("predecessors", []),
        }

    # Validate predecessors
    for tid, t in task_map.items():
        for pred in t["predecessors"]:
            if pred not in task_map:
                return jsonify({"error": f"Task '{tid}' has unknown predecessor '{pred}'."}), 400

    # Topological sort (Kahn's algorithm)
    in_degree = defaultdict(int)
    successors = defaultdict(list)
    for tid, t in task_map.items():
        in_degree.setdefault(tid, 0)
        for pred in t["predecessors"]:
            successors[pred].append(tid)
            in_degree[tid] += 1

    queue = deque([tid for tid in task_map if in_degree[tid] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != len(task_map):
        return jsonify({"error": "Cycle detected in task dependencies."}), 400

    # Forward pass — Early Start (ES) and Early Finish (EF)
    es = {}
    ef = {}
    for tid in topo_order:
        t = task_map[tid]
        if t["predecessors"]:
            es[tid] = max(ef[p] for p in t["predecessors"])
        else:
            es[tid] = 0
        ef[tid] = es[tid] + t["duration"]

    project_duration = max(ef.values()) if ef else 0

    # Backward pass — Late Finish (LF) and Late Start (LS)
    lf = {}
    ls = {}
    end_tasks = [tid for tid in task_map if not successors[tid]]
    for tid in end_tasks:
        lf[tid] = project_duration

    for tid in reversed(topo_order):
        if tid not in lf:
            lf[tid] = min(ls[s] for s in successors[tid])
        ls[tid] = lf[tid] - task_map[tid]["duration"]

    # Float and critical path
    results = []
    critical_path = []
    for tid in topo_order:
        total_float = ls[tid] - es[tid]
        is_critical = abs(total_float) < 0.0001
        if is_critical:
            critical_path.append(tid)
        results.append({
            "id": tid,
            "name": task_map[tid]["name"],
            "duration": task_map[tid]["duration"],
            "es": round(es[tid], 2),
            "ef": round(ef[tid], 2),
            "ls": round(ls[tid], 2),
            "lf": round(lf[tid], 2),
            "total_float": round(total_float, 2),
            "critical": is_critical,
        })

    return jsonify({
        "project_duration": round(project_duration, 2),
        "critical_path": critical_path,
        "tasks": results,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
