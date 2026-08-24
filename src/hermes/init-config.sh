#!/bin/sh
set -eu

config_path=/opt/data/config.yaml

if [ ! -s "$config_path" ]; then
  cp /opt/hermes-config.yaml "$config_path"
fi

exec /opt/hermes/docker/entrypoint-dispatch.sh "$@"
