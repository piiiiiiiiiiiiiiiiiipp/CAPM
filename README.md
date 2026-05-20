<<<<<<< HEAD
# CAPM
=======
# CAPM — Cloud-Agile Performance Monitor
**Dissertation Project — Khanafiyeva Elnara, 2025**
*Evaluating the Impact of Cloud Technologies on IT Project Management Efficiency*

---

## Quick Start (Docker — recommended)

```bash
# 1. Clone / download the project
cd capm

# 2. Start everything (DB + API + Frontend)
docker-compose up --build

# 3. Open browser
open http://localhost
```

That's it. The full stack is running.

---

## What's Running

| Service  | URL                          | Description                  |
|----------|------------------------------|------------------------------|
| Frontend | http://localhost             | Login, Dashboard, Landing    |
| API      | http://localhost/api         | FastAPI backend               |
| API Docs | http://localhost/docs        | Swagger UI                   |
| Database | localhost:5432               | PostgreSQL (internal)        |

---

## Architecture

```
Browser
  │
  ▼
Nginx :80 ────────── /api/* ──────► FastAPI :8000
  │                                      │
  │ static files                         │
  ▼                                      ▼
frontend/                           PostgreSQL :5432
  ├── index.html    (Landing)            │
  ├── login.html    (Auth)         ML Models (.pkl)
  └── dashboard.html (App)          ├── maturity_classifier.pkl
                                    ├── ceiling_predictor.pkl
                                    └── hes_forecaster.pkl
```

---

## CAEM Metrics (from dissertation)

| Metric | Formula | Target |
|--------|---------|--------|
| H_es   | (V × E_l) / (C + D_k) × 100 | > 80 |
| D_k    | D_0 × e^(α × O_c)           | < 30 |
| O_c    | log(microservices × deps) × (latency/100) | < 3.5 |
| E_l    | (used/allocated) × (1 + 0.05 × scaling) | — |
| R2V    | cloud_cost / deployments_per_sprint | < $1,200 |
| CASI   | w₁×CEI + w₂×SVS + w₃×RRE | > 0.70 |

---

## ML Models

| Model | Algorithm | Target | Accuracy |
|-------|-----------|--------|----------|
| Maturity Classifier | Random Forest (200 trees) | L1-L5 maturity level | ~93% |
| Complexity Ceiling Predictor | Gradient Boosting | Will hit O_c > 4.5 in 5 sprints? | ~89% |
| H_es Forecaster | Gradient Boosting Regressor | H_es next sprint | MAE ~4.2 |

Trained on 3,000 synthetic profiles derived from empirical parameters of 47 real IT projects.

---

## Manual Setup (without Docker)

### Backend
```bash
cd backend
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/capm"
export SECRET_KEY="your-secret-key"

# Run
uvicorn main:app --reload --port 8000
```

### Frontend
Just open `frontend/index.html` in a browser.
For local dev without Docker, set `API = 'http://localhost:8000'` in the HTML files.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql+asyncpg://capm_user:capm_password_change_me@db:5432/capm | PostgreSQL connection |
| SECRET_KEY | dev-secret | JWT signing key — **change in production!** |
| MODEL_DIR | /app/models | Directory for ML model files |

---

## Production Checklist

- [ ] Change `POSTGRES_PASSWORD` in `docker-compose.yml`
- [ ] Change `SECRET_KEY` in `docker-compose.yml`
- [ ] Set up SSL/HTTPS (add Certbot or load balancer)
- [ ] Set `allow_origins` in `main.py` to your domain only
- [ ] Set up DB backups (`docker exec capm_db pg_dump ...`)

---

## API Endpoints

```
POST   /auth/register              Register new user
POST   /auth/login                 Login → JWT token
GET    /auth/me                    Current user

GET    /projects                   List projects
POST   /projects                   Create project
GET    /projects/{id}              Get project
DELETE /projects/{id}              Delete project

POST   /projects/{id}/metrics      Save + compute all CAEM metrics + ML
GET    /projects/{id}/metrics      Full metrics history
GET    /projects/{id}/metrics/latest  Latest snapshot

POST   /predict                    Real-time ML prediction (no DB save)

GET    /analytics/summary          Cross-project summary
GET    /analytics/model-info       ML model metadata

GET    /health                     Health check
```

All protected routes require: `Authorization: Bearer <token>`
>>>>>>> master
