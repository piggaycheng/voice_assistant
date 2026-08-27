#!/bin/sh
set -eu

config_path=/opt/data/config.yaml

cp /opt/hermes-config.yaml "$config_path"

exec /opt/hermes/docker/entrypoint-dispatch.sh "$@"
