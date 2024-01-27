CALL_CMD=PYTHONPATH=. python
CONFIG_PATH=configs/base_config.yaml
ACTIVATE_VENV=source .venv/bin/activate

SHELL := /bin/bash
.ONESHELL:

setup:
	python -m venv .venv
	$(ACTIVATE_VENV)

	pip install -r requirements.txt
	dvc install
	dvc pull
	clearml-init

train:
	$(ACTIVATE_VENV)
	$(CALL_CMD) src/train.py $(CONFIG_PATH)