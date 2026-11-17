FROM python:3.11-slim
WORKDIR /app
# torch CPU-only evita baixar ~2GB de CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG EMBED_MODEL=PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir
ENV EMBED_MODEL=${EMBED_MODEL} CHROMA_DIR=/data/chroma DOC_DIR=/data/doc
# baixa o modelo de embedding no build (container sobe sem depender da internet)
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBED_MODEL}')"
COPY app/ .
