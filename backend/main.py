"""
CAPM — Cloud-Agile Performance Monitor
FastAPI Backend with PostgreSQL + ML
Dissertation: Khanafiyeva Elnara, 2025
"""

import os, json, math, time, hashlib, hmac, base64
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime, ForeignKey, func, select, text

from ml_predictor import predict_all, train_models, calc_hybrid_efficiency_score, \
    calc_economic_drag, calc_complexity_index, calc_elasticity_factor, calc_r2v, calc_casi

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://capm_user:capm_password_change_me@localhost:5432/capm")
# Fix for SQLAlchemy async: ensure +asyncpg driver
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

SECRET_KEY  = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
TOKEN_TTL   = 60 * 60 * 24 * 7  # 7 days

# ─────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id:         Mapped[int]           = mapped_column(Integer, primary_key=True)
    email:      Mapped[str]           = mapped_column(String(255), unique=True, nullable=False)
    name:       Mapped[str]           = mapped_column(String(255), nullable=False)
    password:   Mapped[str]           = mapped_column(String(255), nullable=False)
    company:    Mapped[Optional[str]] = mapped_column(String(255))
    role:       Mapped[str]           = mapped_column(String(50), default="user")
    created_at: Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)

class Project(Base):
    __tablename__ = "projects"
    id:           Mapped[int]           = mapped_column(Integer, primary_key=True)
    user_id:      Mapped[int]           = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name:         Mapped[str]           = mapped_column(String(255), nullable=False)
    type:         Mapped[str]           = mapped_column(String(100), nullable=False)
    cloud:        Mapped[str]           = mapped_column(String(50), nullable=False)
    team_size:    Mapped[int]           = mapped_column(Integer, default=1)
    baseline:     Mapped[str]           = mapped_column(String(50), default="onprem")
    description:  Mapped[Optional[str]] = mapped_column(Text)
    created_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SprintMetrics(Base):
    __tablename__ = "sprint_metrics"
    id:           Mapped[int]           = mapped_column(Integer, primary_key=True)
    project_id:   Mapped[int]           = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    period:       Mapped[str]           = mapped_column(String(20), nullable=False)
    sprint_num:   Mapped[int]           = mapped_column(Integer, nullable=False)
    # Raw inputs
    microservices:   Mapped[int]    = mapped_column(Integer, default=5)
    team_size_snap:  Mapped[int]    = mapped_column(Integer, default=8)
    cloud_maturity:  Mapped[float]  = mapped_column(Float, default=50.0)
    iac_adopted:     Mapped[bool]   = mapped_column(Boolean, default=False)
    deployment_freq: Mapped[float]  = mapped_column(Float, default=2.0)
    avg_latency_ms:  Mapped[float]  = mapped_column(Float, default=200.0)
    dependencies:    Mapped[int]    = mapped_column(Integer, default=5)
    monthly_cost:    Mapped[float]  = mapped_column(Float, nullable=False)
    velocity:        Mapped[float]  = mapped_column(Float, nullable=False)
    costs_json:      Mapped[str]    = mapped_column(Text, default="[]")
    workload_json:   Mapped[str]    = mapped_column(Text, default="[]")
    velocities_json: Mapped[str]    = mapped_column(Text, default="[]")
    rre_pct:         Mapped[float]  = mapped_column(Float, default=75.0)
    # CAEM formula metrics
    Hes:    Mapped[float] = mapped_column(Float)
    Dk:     Mapped[float] = mapped_column(Float)
    Oc:     Mapped[float] = mapped_column(Float)
    El:     Mapped[float] = mapped_column(Float)
    R2V:    Mapped[float] = mapped_column(Float)
    CASI:   Mapped[float] = mapped_column(Float)
    SVS:    Mapped[float] = mapped_column(Float)
    CEI:    Mapped[float] = mapped_column(Float)
    RRE:    Mapped[float] = mapped_column(Float)
    # ML outputs
    ml_maturity:       Mapped[int]   = mapped_column(Integer)
    ml_ceiling_risk:   Mapped[int]   = mapped_column(Integer, default=0)
    ml_ceiling_prob:   Mapped[float] = mapped_column(Float, default=0.0)
    ml_hes_next:       Mapped[float] = mapped_column(Float, default=0.0)
    maturity_rule:     Mapped[int]   = mapped_column(Integer, default=1)
    alerts_json:       Mapped[str]   = mapped_column(Text, default="[]")
    ai_insights_json:  Mapped[Optional[str]] = mapped_column(Text)
    w1: Mapped[float] = mapped_column(Float, default=0.35)
    w2: Mapped[float] = mapped_column(Float, default=0.40)
    w3: Mapped[float] = mapped_column(Float, default=0.25)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────────
# DB SETUP
# ─────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ─────────────────────────────────────────────
# AUTH (JWT without external deps)
# ─────────────────────────────────────────────
bearer = HTTPBearer()

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))

def create_token(user_id: int, email: str) -> str:
    header  = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = b64url(json.dumps({"sub": user_id, "email": email, "exp": int(time.time()) + TOKEN_TTL}).encode())
    sig = hmac.new(SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{b64url(sig)}"

def verify_token(token: str) -> dict:
    try:
        h, p, s = token.split(".")
        expected = b64url(hmac.new(SECRET_KEY.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, s):
            raise ValueError("bad sig")
        payload = json.loads(b64url_decode(p))
        if payload["exp"] < time.time():
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def hash_pw(pw: str) -> str:
    return hashlib.sha256((pw + SECRET_KEY).encode()).hexdigest()

def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    return verify_token(creds.credentials)

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RegisterIn(BaseModel):
    email: str; name: str; password: str; company: Optional[str] = None

class LoginIn(BaseModel):
    email: str; password: str

class ProjectIn(BaseModel):
    name: str; type: str; cloud: str
    team_size: int = 8; baseline: str = "onprem"
    description: Optional[str] = None

class MetricsIn(BaseModel):
    period: str; sprint_num: int
    # Raw CAEM inputs
    microservices:   int   = 5
    cloud_maturity:  float = 50.0
    iac_adopted:     bool  = False
    deployment_freq: float = 2.0
    avg_latency_ms:  float = 200.0
    dependencies:    int   = 5
    monthly_cost:    float
    velocity:        float
    # Array data for chart metrics
    costs:      List[float] = []
    workload:   List[float] = []
    velocities: List[float] = []
    rre_pct:    float = 75.0
    # Weights
    w1: float = 0.35; w2: float = 0.40; w3: float = 0.25
    # Optional
    ai_insights: Optional[str] = None

class PredictIn(BaseModel):
    sprint_num: int; microservices: int; team_size: int
    cloud_maturity: float; iac_adopted: bool
    deployment_freq: float; avg_latency_ms: float
    dependencies: int; monthly_cost: float; velocity: float
    w1: float = 0.35; w2: float = 0.40; w3: float = 0.25

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Train ML models (background, non-blocking)
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, train_models)
    print("✅ CAPM API ready")
    yield

app = FastAPI(
    title="CAPM API",
    description="Cloud-Agile Performance Monitor — Khanafiyeva Elnara Dissertation 2025",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
async def register(body: RegisterIn, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email.lower().strip()))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")
    user = User(email=body.email.lower().strip(), name=body.name.strip(),
                password=hash_pw(body.password), company=body.company)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"token": create_token(user.id, user.email),
            "user":  {"id": user.id, "email": user.email, "name": user.name, "company": user.company}}

@app.post("/auth/login")
async def login(body: LoginIn, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower().strip()))
    user = result.scalar_one_or_none()
    if not user or user.password != hash_pw(body.password):
        raise HTTPException(401, "Invalid email or password")
    return {"token": create_token(user.id, user.email),
            "user":  {"id": user.id, "email": user.email, "name": user.name, "company": user.company}}

@app.get("/auth/me")
async def me(user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user["sub"]))
    u = result.scalar_one_or_none()
    if not u:
        raise HTTPException(404, "User not found")
    return {"id": u.id, "email": u.email, "name": u.name, "company": u.company, "created_at": u.created_at.isoformat()}

# ─────────────────────────────────────────────
# PROJECT ROUTES
# ─────────────────────────────────────────────
@app.get("/projects")
async def list_projects(user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Project).where(Project.user_id == user["sub"]).order_by(Project.updated_at.desc())
    )
    projects = result.scalars().all()
    out = []
    for p in projects:
        # Get latest metrics
        m_res = await db.execute(
            select(SprintMetrics)
            .where(SprintMetrics.project_id == p.id)
            .order_by(SprintMetrics.created_at.desc())
            .limit(1)
        )
        latest = m_res.scalar_one_or_none()
        d = {"id": p.id, "name": p.name, "type": p.type, "cloud": p.cloud,
             "team_size": p.team_size, "baseline": p.baseline, "description": p.description,
             "created_at": p.created_at.isoformat(), "updated_at": p.updated_at.isoformat()}
        if latest:
            d["latest"] = {"Hes": latest.Hes, "CASI": latest.CASI, "Oc": latest.Oc,
                           "maturity": latest.ml_maturity, "ceiling_risk": latest.ml_ceiling_risk}
        out.append(d)
    return out

@app.post("/projects", status_code=201)
async def create_project(body: ProjectIn, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    p = Project(user_id=user["sub"], name=body.name, type=body.type, cloud=body.cloud,
                team_size=body.team_size, baseline=body.baseline, description=body.description)
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return {"id": p.id, "name": p.name, "type": p.type, "cloud": p.cloud,
            "team_size": p.team_size, "created_at": p.created_at.isoformat()}

@app.get("/projects/{pid}")
async def get_project(pid: int, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == pid, Project.user_id == user["sub"]))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Project not found")
    return {"id": p.id, "name": p.name, "type": p.type, "cloud": p.cloud,
            "team_size": p.team_size, "baseline": p.baseline, "description": p.description}

@app.delete("/projects/{pid}", status_code=204)
async def delete_project(pid: int, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == pid, Project.user_id == user["sub"]))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Project not found")
    await db.execute(text("DELETE FROM sprint_metrics WHERE project_id = :pid"), {"pid": pid})
    await db.delete(p)
    await db.commit()

# ─────────────────────────────────────────────
# METRICS ROUTES
# ─────────────────────────────────────────────
@app.post("/projects/{pid}/metrics", status_code=201)
async def save_metrics(pid: int, body: MetricsIn, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    # Verify ownership
    res = await db.execute(select(Project).where(Project.id == pid, Project.user_id == user["sub"]))
    if not res.scalar_one_or_none():
        raise HTTPException(404, "Project not found")

    # Full CAEM computation (formulas + ML)
    result = predict_all(
        sprint_num=body.sprint_num, microservices=body.microservices,
        team_size=body.team_size if hasattr(body, 'team_size') else 8,
        cloud_maturity=body.cloud_maturity, iac_adopted=body.iac_adopted,
        deployment_freq=body.deployment_freq, avg_latency_ms=body.avg_latency_ms,
        dependencies=body.dependencies, monthly_cost=body.monthly_cost,
        velocity=body.velocity, w1=body.w1, w2=body.w2, w3=body.w3
    )

    m = SprintMetrics(
        project_id=pid, period=body.period, sprint_num=body.sprint_num,
        microservices=body.microservices, cloud_maturity=body.cloud_maturity,
        iac_adopted=body.iac_adopted, deployment_freq=body.deployment_freq,
        avg_latency_ms=body.avg_latency_ms, dependencies=body.dependencies,
        monthly_cost=body.monthly_cost, velocity=body.velocity,
        costs_json=json.dumps(body.costs), workload_json=json.dumps(body.workload),
        velocities_json=json.dumps(body.velocities), rre_pct=body.rre_pct,
        Hes=result["Hes"], Dk=result["Dk"], Oc=result["Oc"], El=result["El"],
        R2V=result["R2V"], CASI=result["CASI"], SVS=result["SVS"],
        CEI=result["CEI"], RRE=result["RRE"],
        ml_maturity=result["ml"]["maturity_level"],
        ml_ceiling_risk=result["ml"]["ceiling_risk"],
        ml_ceiling_prob=result["ml"]["ceiling_prob"],
        ml_hes_next=result["ml"]["hes_next_sprint"],
        maturity_rule=result["maturity_rule"],
        alerts_json=json.dumps(result["alerts"]),
        ai_insights_json=body.ai_insights,
        w1=body.w1, w2=body.w2, w3=body.w3
    )
    db.add(m)

    # Update project timestamp
    await db.execute(text("UPDATE projects SET updated_at=NOW() WHERE id=:pid"), {"pid": pid})
    await db.commit()
    await db.refresh(m)

    return {**result, "id": m.id, "project_id": pid, "period": body.period,
            "sprint_num": body.sprint_num, "created_at": m.created_at.isoformat()}

@app.get("/projects/{pid}/metrics")
async def get_metrics(pid: int, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Project).where(Project.id == pid, Project.user_id == user["sub"]))
    if not res.scalar_one_or_none():
        raise HTTPException(404, "Project not found")
    rows = await db.execute(
        select(SprintMetrics).where(SprintMetrics.project_id == pid).order_by(SprintMetrics.sprint_num)
    )
    out = []
    for m in rows.scalars().all():
        out.append({
            "id": m.id, "period": m.period, "sprint_num": m.sprint_num,
            "Hes": m.Hes, "Dk": m.Dk, "Oc": m.Oc, "El": m.El,
            "R2V": m.R2V, "CASI": m.CASI, "SVS": m.SVS, "CEI": m.CEI, "RRE": m.RRE,
            "ml_maturity": m.ml_maturity, "ml_ceiling_risk": m.ml_ceiling_risk,
            "ml_ceiling_prob": m.ml_ceiling_prob, "ml_hes_next": m.ml_hes_next,
            "alerts": json.loads(m.alerts_json or "[]"),
            "monthly_cost": m.monthly_cost, "velocity": m.velocity,
            "Oc_warning": m.Oc >= 3.5, "Oc_critical": m.Oc >= 4.5,
            "created_at": m.created_at.isoformat()
        })
    return out

@app.get("/projects/{pid}/metrics/latest")
async def get_latest(pid: int, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Project).where(Project.id == pid, Project.user_id == user["sub"]))
    if not res.scalar_one_or_none():
        raise HTTPException(404, "Project not found")
    row = await db.execute(
        select(SprintMetrics).where(SprintMetrics.project_id == pid)
        .order_by(SprintMetrics.created_at.desc()).limit(1)
    )
    m = row.scalar_one_or_none()
    if not m:
        raise HTTPException(404, "No metrics saved yet")
    return {
        "Hes": m.Hes, "Dk": m.Dk, "Oc": m.Oc, "El": m.El, "R2V": m.R2V,
        "CASI": m.CASI, "SVS": m.SVS, "CEI": m.CEI, "RRE": m.RRE,
        "ml": {"maturity_level": m.ml_maturity, "ceiling_risk": m.ml_ceiling_risk,
               "ceiling_prob": m.ml_ceiling_prob, "hes_next_sprint": m.ml_hes_next},
        "maturity_rule": m.maturity_rule,
        "alerts": json.loads(m.alerts_json or "[]"),
        "sprint_num": m.sprint_num, "period": m.period
    }

# ─────────────────────────────────────────────
# ML PREDICT (standalone — no project needed)
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict_endpoint(body: PredictIn, user=Depends(current_user)):
    """
    Direct ML prediction without saving to DB.
    Use for real-time what-if analysis.
    """
    result = predict_all(
        sprint_num=body.sprint_num, microservices=body.microservices,
        team_size=body.team_size, cloud_maturity=body.cloud_maturity,
        iac_adopted=body.iac_adopted, deployment_freq=body.deployment_freq,
        avg_latency_ms=body.avg_latency_ms, dependencies=body.dependencies,
        monthly_cost=body.monthly_cost, velocity=body.velocity,
        w1=body.w1, w2=body.w2, w3=body.w3
    )
    return result

# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
@app.get("/analytics/summary")
async def analytics_summary(user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Project).where(Project.user_id == user["sub"]))
    projects = res.scalars().all()
    if not projects:
        return {"total_projects": 0, "avg_Hes": None, "avg_CASI": None, "alerts_total": 0}

    stats = {"total_projects": len(projects), "ceiling_warnings": 0, "total_alerts": 0,
             "hes_vals": [], "casi_vals": [], "oc_vals": []}

    for p in projects:
        row = await db.execute(
            select(SprintMetrics).where(SprintMetrics.project_id == p.id)
            .order_by(SprintMetrics.created_at.desc()).limit(1)
        )
        m = row.scalar_one_or_none()
        if m:
            stats["hes_vals"].append(m.Hes)
            stats["casi_vals"].append(m.CASI)
            stats["oc_vals"].append(m.Oc)
            if m.ml_ceiling_risk: stats["ceiling_warnings"] += 1
            stats["total_alerts"] += len(json.loads(m.alerts_json or "[]"))

    def safe_mean(lst): return round(sum(lst)/len(lst), 2) if lst else None

    return {
        "total_projects":   stats["total_projects"],
        "avg_Hes":          safe_mean(stats["hes_vals"]),
        "avg_CASI":         safe_mean(stats["casi_vals"]),
        "avg_Oc":           safe_mean(stats["oc_vals"]),
        "ceiling_warnings": stats["ceiling_warnings"],
        "total_alerts":     stats["total_alerts"],
        "projects_at_risk": sum(1 for o in stats["oc_vals"] if o >= 3.5)
    }

@app.get("/analytics/model-info")
async def model_info(user=Depends(current_user)):
    """Return ML model metadata for dashboard display"""
    return {
        "models": [
            {"name": "Maturity Classifier", "type": "Random Forest", "n_estimators": 200,
             "target": "Elastic Agile Maturity Level (L1-L5)", "accuracy": "~93%",
             "trained_on": "3000 synthetic project profiles (dissertation empirical parameters)"},
            {"name": "Complexity Ceiling Predictor", "type": "Gradient Boosting Classifier",
             "target": "Binary: will hit O_c > 4.5 within 5 sprints?", "accuracy": "~89%",
             "trained_on": "Sprint 24 inflection point data (47 real projects)"},
            {"name": "H_es Forecaster", "type": "Gradient Boosting Regressor",
             "target": "H_es score for next sprint", "mae": "~4.2 points",
             "trained_on": "Time-series sprint efficiency trajectories"},
        ],
        "dissertation_params": {
            "projects_analyzed": 47,
            "sprints_total": 1200,
            "inflection_sprint": 24,
            "efficiency_drop_at_ceiling": "42%",
            "R2_maturity_vs_performance": 0.94,
            "iac_cost_improvement": "22%",
            "iac_velocity_improvement": "31%"
        }
    }

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "app": "CAPM API v2", "docs": "/docs"}

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "healthy" if db_ok else "degraded",
            "database": "connected" if db_ok else "error",
            "time": datetime.utcnow().isoformat()}
