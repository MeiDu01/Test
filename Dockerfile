
# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.8-alpine AS builder

RUN apk add python3-dev g++ gcc

WORKDIR /src
RUN pip3 install zstandard
COPY pip-requirements.txt /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r pip-requirements.txt

COPY . .
WORKDIR /src/src
CMD ["python3", "server.py"]

FROM builder as dev-envs

CMD ["python3", "-m" , "flask", "--app", "web_service.py", "--debug", "run", "--host=0.0.0.0", "--port=5008"]