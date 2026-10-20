# ChemPrompt Web Service image — shared by the api, init and aya services.
# Dependencies come from the same environment.yml as the local conda env (single source of versions).
FROM mambaorg/micromamba:1.5.10

COPY --chown=$MAMBA_USER:$MAMBA_USER config/environment.yml /tmp/environment.yml
# no GPU at build time: tell the solver which CUDA the CUDA-enabled pytorch build targets
RUN CONDA_OVERRIDE_CUDA=12.1 micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend MODEL_STORAGE_ROOT=/data/storage

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER sql /app/sql
COPY --chown=$MAMBA_USER:$MAMBA_USER backend /app/backend
COPY --chown=$MAMBA_USER:$MAMBA_USER aya_service /app/aya_service
COPY bin/docker-entrypoint.sh /usr/local/bin/chemprompt-entrypoint.sh
COPY bin/chemprompt.sh /usr/local/bin/chemprompt
USER root
RUN mkdir -p /data/storage && chown -R $MAMBA_USER:$MAMBA_USER /data

# starts as root only to look up the owner of the mounted storage folder, then drops to that user
WORKDIR /app/backend
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/chemprompt-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
