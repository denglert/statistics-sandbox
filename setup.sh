#!/usr/bin/env bash

BASEDIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export STATISTICS_SANDBOX_DIR=${BASEDIR}

PYTHON_MODULES_DIR=${STATISTICS_SANDBOX_DIR}/python/modules/
export PYTHONPATH=${PYTHON_MODULES_DIR}:${PYTHONPATH}

# - Source toolbox
source ${STATISTICS_SANDBOX_DIR}/toolbox/setup.sh
