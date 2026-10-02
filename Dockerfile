# Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: ENV MODEL_PATH points at the
# MOPD checkpoint, which download-ms.sh fetches from ModelScope. Derived from
# MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see LICENSE and the
# "License & attribution" section of README.md.
#
# MiMo-V2.6 stores its 256 routed experts as MXFP4 (U8 nibble pairs plus E8M0
# block scales). SGLang keeps that layout on the GPU only since 2026-09-20
# (sgl-project/sglang#40448, first in the 2026-09-21 nightly). An older base
# allocates the experts as FP8, ~151 GiB per rank at EP=2, and takes the Spark
# down with it. start.sh passes BASE_IMAGE from the profile; boot.py refuses to
# launch on an engine that lacks the packed layout.
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
# Optional build-time mirror for the one pip install below (restricted networks).
ARG PIP_INDEX_URL=https://pypi.org/simple/
LABEL mimo26.base="${BASE_IMAGE}"
# Image, video, and audio. The MiMo processor decodes audio with torchcodec
# and shells out to ffprobe for video.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" torchcodec \
 && python3 -c "import torchaudio; from torchcodec.decoders import AudioDecoder; import shutil; assert shutil.which('ffprobe'), 'ffprobe missing'"
WORKDIR /opt/mimo26
COPY boot.py /opt/mimo26/boot.py
ENV MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD \
    STATE_PATH=/state \
    OFFLOAD_MODE=off \
    PYTHONUNBUFFERED=1
EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=30s --start-period=40m --retries=3 \
    CMD ["python3", "-S", "/opt/mimo26/boot.py", "health"]
ENTRYPOINT ["python3", "-u", "/opt/mimo26/boot.py"]
CMD ["run"]
