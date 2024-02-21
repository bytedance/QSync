#!/bin/bash
DOCKER_BINARY="docker"
IMAGE_NAME='qsync'

DOCKER_BUILDKIT=1 ${DOCKER_BINARY} build -f Dockerfile -t ${IMAGE_NAME}:latest .