# ============================================================================ #
# Tech Challenge - Fase 3 (FIAP MLET)
# Imagem de producao da aplicacao Streamlit para deploy no EasyPanel.
#
# Base: Python 3.12 (compativel com as versoes fixadas em requirements.txt).
# ============================================================================ #

FROM python:3.12-slim

# --- Variaveis de ambiente (boas praticas para container Python) ----------- #
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Diretorio de trabalho (requisito: projeto copiado para /app) ---------- #
WORKDIR /app


# --- Dependencias (camada separada para melhor cache no rebuild) ----------- #
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Codigo + artefatos (models/, data/, reports/, src/, app/, etc.) ------- #
COPY . .

# --- Porta exposta (requisito: 8501) ---------------------------------------- #
EXPOSE 8501

# --- Streamlit: 0.0.0.0, porta 8501, headless, sem coleta de estatisticas -- #
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
