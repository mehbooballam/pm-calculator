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


@app.route("/api/financial", methods=["POST"])
def calculate_financial():
    """Calculate Payback, NPV, IRR, BCR, ROI, PI, and Break-Even."""
    data = request.json
    try:
        initial_investment = float(data["initial_investment"])
        discount_rate = float(data["discount_rate"])
        cash_flows = [float(c) for c in data["cash_flows"]]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Provide initial_investment, discount_rate, and cash_flows array."}), 400

    if initial_investment < 0:
        return jsonify({"error": "Initial investment must be non-negative."}), 400
    if not cash_flows:
        return jsonify({"error": "Provide at least one cash flow period."}), 400

    r = discount_rate / 100
    n = len(cash_flows)

    dcf = [cf / (1 + r) ** (t + 1) if r != 0 else cf for t, cf in enumerate(cash_flows)]

    npv = -initial_investment + sum(dcf)

    cum_cf, cum_dcf = [], []
    running, running_d = -initial_investment, -initial_investment
    for i in range(n):
        running += cash_flows[i]
        running_d += dcf[i]
        cum_cf.append(round(running, 2))
        cum_dcf.append(round(running_d, 2))

    payback = None
    for t in range(n):
        if cum_cf[t] >= 0:
            if t == 0:
                payback = round(initial_investment / cash_flows[0], 2) if cash_flows[0] > 0 else 0
            else:
                prev = cum_cf[t - 1]
                payback = round(t + abs(prev) / cash_flows[t], 2) if cash_flows[t] > 0 else t + 1
            break

    disc_payback = None
    for t in range(n):
        if cum_dcf[t] >= 0:
            if t == 0:
                disc_payback = round(initial_investment / dcf[0], 2) if dcf[0] > 0 else 0
            else:
                prev = cum_dcf[t - 1]
                disc_payback = round(t + abs(prev) / dcf[t], 2) if dcf[t] > 0 else t + 1
            break

    def npv_at(rate):
        return -initial_investment + sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))

    irr = None
    lo, hi = -0.5, 5.0
    try:
        if npv_at(lo) * npv_at(hi) < 0:
            for _ in range(200):
                mid = (lo + hi) / 2
                if abs(npv_at(mid)) < 0.01:
                    irr = round(mid * 100, 4)
                    break
                if npv_at(lo) * npv_at(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            if irr is None:
                irr = round((lo + hi) / 2 * 100, 4)
    except (ZeroDivisionError, OverflowError):
        irr = None

    total_returns = sum(cash_flows)
    roi = round(((total_returns - initial_investment) / initial_investment) * 100, 4) if initial_investment > 0 else None

    pv_benefits = sum(d for d in dcf if d > 0)
    pv_costs = initial_investment + abs(sum(d for d in dcf if d < 0))
    bcr = round(pv_benefits / pv_costs, 4) if pv_costs > 0 else None

    pv_future = sum(dcf)
    pi = round(pv_future / initial_investment, 4) if initial_investment > 0 else None

    fixed_costs = float(data.get("fixed_costs", 0))
    variable_cost = float(data.get("variable_cost_per_unit", 0))
    selling_price = float(data.get("selling_price_per_unit", 0))
    bep_units = bep_dollars = contribution_margin = None
    if selling_price > 0 and selling_price > variable_cost:
        contribution_margin = round(selling_price - variable_cost, 2)
        bep_units = round(fixed_costs / contribution_margin, 2) if contribution_margin > 0 else None
        cm_ratio = contribution_margin / selling_price
        bep_dollars = round(fixed_costs / cm_ratio, 2) if cm_ratio > 0 else None

    npv_sensitivity = []
    for test_rate in range(0, 51, 2):
        tr = test_rate / 100
        test_npv = -initial_investment + sum(cf / (1 + tr) ** (t + 1) for t, cf in enumerate(cash_flows))
        npv_sensitivity.append({"rate": test_rate, "npv": round(test_npv, 2)})

    return jsonify({
        "npv": round(npv, 2), "irr": irr, "roi": roi, "bcr": bcr, "pi": pi,
        "payback": payback, "disc_payback": disc_payback,
        "cash_flows": [round(c, 2) for c in cash_flows],
        "dcf": [round(d, 2) for d in dcf],
        "cum_cf": cum_cf, "cum_dcf": cum_dcf,
        "total_returns": round(total_returns, 2),
        "initial_investment": round(initial_investment, 2),
        "discount_rate": discount_rate,
        "bep_units": bep_units, "bep_dollars": bep_dollars,
        "contribution_margin": contribution_margin,
        "fixed_costs": fixed_costs,
        "variable_cost_per_unit": variable_cost,
        "selling_price_per_unit": selling_price,
        "npv_sensitivity": npv_sensitivity,
        "pv_benefits": round(pv_benefits, 2),
        "pv_costs": round(pv_costs, 2),
    })


@app.route("/api/emv", methods=["POST"])
def calculate_emv():
    """Calculate Expected Monetary Value for project risks."""
    data = request.json
    risks = data.get("risks", [])

    if not risks:
        return jsonify({"error": "Provide at least one risk item."}), 400

    items = []
    total_emv = 0
    total_threats = 0
    total_opportunities = 0

    for r in risks:
        try:
            name = r["name"]
            probability = float(r["probability"])
            impact = float(r["impact"])
        except (KeyError, ValueError, TypeError):
            return jsonify({"error": "Each risk must have name, probability (0-1), and impact."}), 400

        if probability < 0 or probability > 1:
            return jsonify({"error": f"Probability for '{name}' must be between 0 and 1."}), 400

        emv = probability * impact
        total_emv += emv
        if impact < 0:
            total_threats += emv
        else:
            total_opportunities += emv

        items.append({
            "name": name,
            "type": r.get("type", "Threat"),
            "probability": round(probability, 4),
            "impact": round(impact, 2),
            "emv": round(emv, 2),
        })

    return jsonify({
        "items": items,
        "total_emv": round(total_emv, 2),
        "total_threats": round(total_threats, 2),
        "total_opportunities": round(total_opportunities, 2),
        "risk_count": len(items),
        "threat_count": sum(1 for i in items if i["impact"] < 0),
        "opportunity_count": sum(1 for i in items if i["impact"] >= 0),
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
            "predecessors": task_map[tid].get("predecessors", []),
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
