#!/usr/bin/env bash

BASEDIR=$(dirname $(realpath "$BASH_SOURCE"))
export STATISTICS_SANDBOX_DIR=${BASEDIR}

PYTHON_MODULES_DIR=${STATISTICS_SANDBOX_DIR}/python/modules/
export PYTHONPATH=${PYTHON_MODULES_DIR}:${PYTHONPATH}

# - Source toolbox
source ${STATISTICS_SANDBOX_DIR}/toolbox/setup.sh
