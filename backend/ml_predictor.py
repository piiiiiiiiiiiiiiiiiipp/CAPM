"""
CAPM ML Predictor Module
Based on dissertation: "Evaluating the Impact of Cloud Technologies on IT Project Management"
Khanafiyeva Elnara, 2025

Models:
1. Complexity Ceiling Classifier — predicts if project will hit the Complexity Ceiling
   within next 5 sprints (threshold O_c > 4.5, seen in 83% of budget-overrun projects)

2. H_es Forecaster — predicts Hybrid Efficiency Score for next 3 sprints
   using Gradient Boosting Regressor

3. Maturity Classifier — classifies Elastic Agile Maturity Level (L1-L5)
   using Random Forest on CAEM metrics
"""

import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# CAEM FORMULAS (from dissertation Section 2 / Section 3)
# ─────────────────────────────────────────────────────────

def calc_economic_drag(D0: float, alpha: float, Oc: float) -> float:
    """
    D_k = D_0 * e^(alpha * O_c)
    D_0 — baseline operational overhead
    alpha — interdependence factor
    O_c — Complexity Index
    Target: D_k < 30
    """
    return D0 * np.exp(alpha * Oc)

def calc_hybrid_efficiency_score(V: float, El: float, C: float, Dk: float) -> float:
    """
    H_es = (V * E_l) / (C + D_k) * 100
    V   — Sprint Velocity
    E_l — Infrastructure Elasticity coefficient (0-1)
    C   — Direct Cloud Consumption costs
    D_k — Economic Drag
    Target: H_es > 80
    """
    denominator = C + Dk
    if denominator <= 0:
        return 0.0
    return min(100.0, (V * El) / denominator * 100)

def calc_elasticity_factor(allocated: float, used: float, scaling_events: int) -> float:
    """
    E_l = (used / allocated) * (1 + 0.1 * scaling_events)
    Measures how efficiently cloud resources are utilized with scaling
    """
    if allocated <= 0:
        return 0.0
    base = used / allocated
    return min(1.0, base * (1 + 0.05 * scaling_events))

def calc_complexity_index(microservices: int, avg_api_latency_ms: float, dependencies: int) -> float:
    """
    O_c = log(microservices * dependencies) * (avg_api_latency_ms / 100)
    Warning threshold: 3.5
    Critical threshold: 4.5 (83% budget overrun probability)
    """
    if microservices <= 0 or dependencies <= 0:
        return 1.0
    return np.log(microservices * dependencies) * (avg_api_latency_ms / 100)

def calc_r2v(cloud_cost: float, deployments_per_sprint: int) -> float:
    """
    R2V = cloud_cost / deployments_per_sprint
    Resource-to-Velocity ratio: cost per deployment
    Target: R2V < $1,200
    """
    if deployments_per_sprint <= 0:
        return cloud_cost
    return cloud_cost / deployments_per_sprint

def calc_casi(CEI: float, SVS: float, RRE: float, w1=0.35, w2=0.40, w3=0.25) -> float:
    """
    CASI = w1*CEI + w2*SVS + w3*RRE  (weights normalized to sum=1)
    Cloud-Agile Synergy Index
    """
    total = w1 + w2 + w3
    return (w1/total)*CEI + (w2/total)*SVS + (w3/total)*RRE

def classify_maturity_rule(Hes: float, Oc: float, Dk: float, CASI: float) -> int:
    """
    Rule-based Elastic Agile Maturity Model (5 levels)
    Based on empirical thresholds from 47 projects analysis
    L1 Static / L2 Reactive / L3 Proactive / L4 Predictive / L5 Autonomous
    """
    if CASI >= 0.85 and Hes >= 85 and Oc < 3.0 and Dk < 20:
        return 5  # Autonomous
    if CASI >= 0.70 and Hes >= 75 and Oc < 3.5 and Dk < 25:
        return 4  # Predictive
    if CASI >= 0.55 and Hes >= 60 and Oc < 4.0 and Dk < 35:
        return 3  # Proactive
    if CASI >= 0.40 and Hes >= 45:
        return 2  # Reactive
    return 1      # Static

# ─────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION (based on dissertation findings)
# ─────────────────────────────────────────────────────────

def generate_training_data(n_samples: int = 2000, random_state: int = 42) -> dict:
    """
    Generate synthetic training data based on dissertation empirical findings:
    - 47 real projects, 1200+ sprints over 4 years
    - R² = 0.94 between maturity and performance
    - Complexity Ceiling at Sprint 24 in 92% of affected projects
    - IaC improves cost predictability +22%, velocity stability +31%
    """
    rng = np.random.RandomState(random_state)

    # Project features
    sprint_num       = rng.randint(1, 50, n_samples)
    microservices    = rng.randint(2, 80, n_samples)
    team_size        = rng.randint(3, 25, n_samples)
    cloud_maturity   = rng.uniform(10, 100, n_samples)
    iac_adopted      = rng.binomial(1, 0.55, n_samples)
    deployment_freq  = rng.uniform(0.5, 8.0, n_samples)
    avg_latency_ms   = rng.uniform(50, 800, n_samples)
    dependencies     = rng.randint(1, 40, n_samples)
    monthly_cost     = rng.uniform(3000, 50000, n_samples)
    velocity         = rng.uniform(20, 80, n_samples)

    # IaC effect: +22% cost predictability, +31% velocity stability
    velocity_boost   = np.where(iac_adopted, velocity * 1.31, velocity)
    cost_reduction   = np.where(iac_adopted, monthly_cost * 0.78, monthly_cost)

    # Complexity Index
    Oc = np.log(np.maximum(microservices * dependencies, 1)) * (avg_latency_ms / 100)
    Oc = np.clip(Oc, 0.5, 8.0)

    # Economic Drag (exponential with complexity)
    D0    = 0.1 * cost_reduction
    alpha = 0.15
    Dk    = D0 * np.exp(alpha * Oc)
    Dk    = np.clip(Dk, 0, 100)

    # Elasticity Factor
    El = np.clip(cloud_maturity/100 * (1 + 0.05*deployment_freq), 0.1, 1.0)

    # H_es: Hybrid Efficiency Score
    Hes = np.clip((velocity_boost * El) / (cost_reduction/10000 + Dk/100) * 100, 0, 100)

    # Sprint Velocity Stability (SVS)
    velocity_std = rng.uniform(2, 15, n_samples)
    SVS = np.clip(1 - velocity_std / velocity_boost, 0, 1)

    # Cost Elasticity Index (CEI)
    cost_change     = rng.uniform(-0.3, 0.5, n_samples)
    workload_change = rng.uniform(0.05, 0.6, n_samples)
    raw_CEI = np.abs(cost_change) / np.maximum(workload_change, 0.01)
    CEI = np.clip(1 - (raw_CEI - 5) / 30, 0, 1)

    # Resource Reallocation Efficiency (RRE)
    RRE = np.clip(cloud_maturity/100 * (0.7 + 0.3*iac_adopted), 0.1, 1.0)

    # CASI
    CASI = 0.35*CEI + 0.40*SVS + 0.25*RRE

    # R2V
    R2V = cost_reduction / np.maximum(deployment_freq * 2, 1)

    # ── TARGETS ──
    # 1. Maturity level (L1-L5)
    maturity_labels = np.array([
        classify_maturity_rule(h, o, d, c)
        for h, o, d, c in zip(Hes, Oc, Dk, CASI)
    ])

    # 2. Complexity Ceiling risk (will hit O_c > 4.5 within 5 sprints?)
    # Sprints 20-25 with high Oc are most at risk (dissertation finding: Sprint 24)
    ceiling_risk = (
        (Oc > 3.5) &
        (sprint_num >= 15) &
        (sprint_num <= 35) &
        (Dk > 30) &
        (microservices > 20)
    ).astype(int)
    # Add noise (not all high-complexity projects hit the ceiling)
    noise_mask = rng.binomial(1, 0.12, n_samples)
    ceiling_risk = np.clip(ceiling_risk + noise_mask, 0, 1)

    # 3. H_es next sprint (regression target)
    Hes_next = np.clip(
        Hes + rng.normal(0, 5, n_samples)
        - np.where(Oc > 3.5, 8, 0)       # complexity drag
        + np.where(iac_adopted, 3, 0)      # IaC benefit
        - np.where(sprint_num >= 24, 4, 0) # inflection point effect
        ,0, 100
    )

    # Features for ML
    X = np.column_stack([
        sprint_num, microservices, team_size, cloud_maturity, iac_adopted,
        deployment_freq, avg_latency_ms, dependencies, Oc, Dk, El,
        Hes, SVS, CEI, RRE, CASI, R2V, monthly_cost, velocity
    ])

    feature_names = [
        'sprint_num', 'microservices', 'team_size', 'cloud_maturity', 'iac_adopted',
        'deployment_freq', 'avg_latency_ms', 'dependencies', 'Oc', 'Dk', 'El',
        'Hes', 'SVS', 'CEI', 'RRE', 'CASI', 'R2V', 'monthly_cost', 'velocity'
    ]

    return {
        'X': X, 'feature_names': feature_names,
        'y_maturity': maturity_labels,
        'y_ceiling': ceiling_risk,
        'y_hes_next': Hes_next
    }

# ─────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────

def train_models(force: bool = False) -> dict:
    """Train all three ML models and save to disk."""

    paths = {
        'maturity':  MODEL_DIR / 'maturity_classifier.pkl',
        'ceiling':   MODEL_DIR / 'ceiling_predictor.pkl',
        'hes':       MODEL_DIR / 'hes_forecaster.pkl',
        'scaler':    MODEL_DIR / 'scaler.pkl',
    }

    if not force and all(p.exists() for p in paths.values()):
        print("✅ ML models already trained, loading from disk")
        return load_models()

    print("🔧 Training ML models on synthetic dissertation data...")
    data = generate_training_data(n_samples=3000)
    X, fn = data['X'], data['feature_names']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, ym_tr, ym_te = train_test_split(X_scaled, data['y_maturity'],  test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X_scaled, data['y_ceiling'],    test_size=0.2, random_state=42)
    _,    _,    yh_tr, yh_te = train_test_split(X_scaled, data['y_hes_next'],   test_size=0.2, random_state=42)

    # 1. Maturity Classifier — Random Forest (interpretable, aligns with dissertation philosophy)
    mat_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    mat_model.fit(X_tr, ym_tr)
    mat_acc = accuracy_score(ym_te, mat_model.predict(X_te))
    print(f"  Maturity Classifier accuracy: {mat_acc:.3f}")

    # 2. Complexity Ceiling Predictor — Gradient Boosting (best for imbalanced threshold detection)
    ceil_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    ceil_model.fit(X_tr, yc_tr)
    ceil_acc = accuracy_score(yc_te, ceil_model.predict(X_te))
    print(f"  Ceiling Predictor accuracy:   {ceil_acc:.3f}")

    # 3. H_es Forecaster — Gradient Boosting Regressor
    hes_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.85, random_state=42
    )
    hes_model.fit(X_tr, yh_tr)
    hes_mae = mean_absolute_error(yh_te, hes_model.predict(X_te))
    print(f"  H_es Forecaster MAE:          {hes_mae:.2f}")

    # Save models
    joblib.dump(scaler,     paths['scaler'])
    joblib.dump(mat_model,  paths['maturity'])
    joblib.dump(ceil_model, paths['ceiling'])
    joblib.dump(hes_model,  paths['hes'])

    print("✅ All models trained and saved")
    return {
        'scaler': scaler, 'maturity': mat_model,
        'ceiling': ceil_model, 'hes': hes_model,
        'metrics': {'maturity_acc': mat_acc, 'ceiling_acc': ceil_acc, 'hes_mae': hes_mae}
    }

def load_models() -> dict:
    paths = {
        'maturity':  MODEL_DIR / 'maturity_classifier.pkl',
        'ceiling':   MODEL_DIR / 'ceiling_predictor.pkl',
        'hes':       MODEL_DIR / 'hes_forecaster.pkl',
        'scaler':    MODEL_DIR / 'scaler.pkl',
    }
    return {k: joblib.load(v) for k, v in paths.items()}

# ─────────────────────────────────────────────────────────
# PREDICTION API
# ─────────────────────────────────────────────────────────

_models = None

def get_models():
    global _models
    if _models is None:
        _models = train_models()
    return _models

def predict_all(
    sprint_num: int, microservices: int, team_size: int,
    cloud_maturity: float, iac_adopted: bool,
    deployment_freq: float, avg_latency_ms: float,
    dependencies: int, monthly_cost: float, velocity: float,
    w1: float = 0.35, w2: float = 0.40, w3: float = 0.25
) -> dict:
    """
    Full CAEM prediction pipeline.
    Returns all 6 metrics + ML predictions.
    """
    # ── Step 1: CAEM formula metrics ──
    Oc = calc_complexity_index(microservices, avg_latency_ms, dependencies)
    Dk = calc_economic_drag(D0=monthly_cost*0.1, alpha=0.15, Oc=Oc)
    El = calc_elasticity_factor(
        allocated=monthly_cost,
        used=monthly_cost * cloud_maturity/100,
        scaling_events=int(deployment_freq)
    )

    vel_boost = velocity * (1.31 if iac_adopted else 1.0)
    cost_eff  = monthly_cost * (0.78 if iac_adopted else 1.0)

    Hes = calc_hybrid_efficiency_score(vel_boost, El, cost_eff/10000, Dk/100)
    R2V = calc_r2v(cost_eff, int(deployment_freq * 2))

    # SVS, CEI, RRE (simplified from available inputs)
    SVS = min(1.0, max(0.0, 1 - (5 / max(velocity, 1))))
    CEI = min(1.0, max(0.0, 1 - (abs(monthly_cost * 0.1) / 30)))
    RRE = min(1.0, cloud_maturity / 100 * (0.7 + 0.3 * int(iac_adopted)))
    CASI = calc_casi(CEI, SVS, RRE, w1, w2, w3)

    maturity_rule = classify_maturity_rule(Hes, Oc, Dk, CASI)

    # ── Step 2: ML predictions ──
    try:
        models = get_models()
        scaler = models['scaler']

        features = np.array([[
            sprint_num, microservices, team_size, cloud_maturity, int(iac_adopted),
            deployment_freq, avg_latency_ms, dependencies, Oc, Dk, El,
            Hes, SVS, CEI, RRE, CASI, R2V, monthly_cost, velocity
        ]])
        X_scaled = scaler.transform(features)

        maturity_ml   = int(models['maturity'].predict(X_scaled)[0])
        maturity_prob = models['maturity'].predict_proba(X_scaled)[0].tolist()

        ceiling_risk  = int(models['ceiling'].predict(X_scaled)[0])
        ceiling_prob  = float(models['ceiling'].predict_proba(X_scaled)[0][1])

        hes_next      = float(models['hes'].predict(X_scaled)[0])

        ml_available = True
    except Exception as e:
        maturity_ml   = maturity_rule
        maturity_prob = []
        ceiling_risk  = 1 if Oc > 3.5 else 0
        ceiling_prob  = min(1.0, Oc / 4.5)
        hes_next      = max(0, Hes - (8 if Oc > 3.5 else 0))
        ml_available  = False

    # ── Step 3: Alerts (dissertation thresholds) ──
    alerts = []
    if Oc >= 4.5:
        alerts.append({"level": "critical", "metric": "O_c", "value": round(Oc, 2),
                       "message": f"Complexity Critical: O_c={Oc:.2f} > 4.5. 83% probability of budget overrun."})
    elif Oc >= 3.5:
        alerts.append({"level": "warning", "metric": "O_c", "value": round(Oc, 2),
                       "message": f"Complexity Warning: O_c={Oc:.2f} > 3.5. 70% chance of hitting ceiling within 5 sprints."})
    if Hes < 60:
        alerts.append({"level": "warning", "metric": "H_es", "value": round(Hes, 1),
                       "message": f"Efficiency below target: H_es={Hes:.1f} (target > 80)."})
    if R2V > 1200:
        alerts.append({"level": "warning", "metric": "R2V", "value": round(R2V, 0),
                       "message": f"High cost-per-deploy: R2V=${R2V:.0f} (target < $1,200)."})
    if Dk > 30:
        alerts.append({"level": "warning", "metric": "D_k", "value": round(Dk, 1),
                       "message": f"Economic Drag elevated: D_k={Dk:.1f} (target < 30)."})
    if sprint_num >= 20 and ceiling_prob > 0.6:
        alerts.append({"level": "critical", "metric": "Ceiling",
                       "value": round(ceiling_prob, 2),
                       "message": f"Complexity Ceiling risk: {ceiling_prob*100:.0f}% probability. Sprint {sprint_num} approaching inflection zone (empirical threshold: Sprint 24)."})

    return {
        # CAEM formula metrics
        "Hes":    round(Hes, 2),
        "Dk":     round(Dk, 2),
        "Oc":     round(Oc, 2),
        "El":     round(El, 4),
        "R2V":    round(R2V, 2),
        "CASI":   round(CASI, 4),
        # Agile metrics
        "SVS":    round(SVS, 4),
        "CEI":    round(CEI, 4),
        "RRE":    round(RRE, 4),
        # ML predictions
        "ml": {
            "available":       ml_available,
            "maturity_level":  maturity_ml,
            "maturity_proba":  [round(p, 3) for p in maturity_prob],
            "ceiling_risk":    ceiling_risk,
            "ceiling_prob":    round(ceiling_prob, 3),
            "hes_next_sprint": round(hes_next, 2),
        },
        # Rule-based maturity
        "maturity_rule": maturity_rule,
        # Alerts
        "alerts": alerts,
        # Status
        "status": "optimized" if Hes >= 80 else "warning" if Hes >= 60 else "critical"
    }
