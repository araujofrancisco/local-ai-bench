# -------------------------------------------------
#  Builder stage – compile the Astro UI
# -------------------------------------------------
FROM node:20-alpine AS builder

# Install the native build tools that esbuild needs
RUN apk add --no-cache python3 make g++

WORKDIR /app

# ---- 1️⃣ Install Node dependencies (cached) ----
COPY web/package*.json ./
# Add --legacy-peer-deps to avoid peer‑dependency issues
RUN npm install --legacy-peer-deps

# ---- 2️⃣ Copy source and build -----------------
COPY web/ .
RUN npm run build   # produces /app/dist

# -------------------------------------------------
#  Backend stage – copy backend code + UI assets
# -------------------------------------------------
FROM python:3.12-slim AS backend

# Install OS‑level build tools (needed for pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 3️⃣ Copy backend source &pyproject.toml ----
COPY src/ ./src/
COPY pyproject.toml .
COPY config/ ./config/

# ---- 4️⃣ Install Python dependencies (editable) ---
RUN pip install --no-cache-dir -e .

# ---- 5️⃣ Copy compiled UI assets into static dir ---
COPY --from=builder /app/dist /app/static

# ---- 6️⃣ Final stage – run the API ---------------
EXPOSE 8000

# Default command runs the FastAPI app
CMD ["uvicorn", "ollama_bench.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
