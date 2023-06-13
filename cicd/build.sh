#!/usr/bin/env bash

set -eou pipefail

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}

docker build -t "$local_repo" .
