PY := .venv-core/bin/python

.PHONY: venv data check repro test graft all
venv:   ; uv venv --python 3.13 .venv-core && VIRTUAL_ENV=.venv-core uv pip install -e ".[dev]"
data:   ; $(PY) scripts/build_dataset.py --root data --start 2015-01-01
check:  ; $(PY) scripts/build_dataset.py --root data --check-only
repro:  ; $(PY) scripts/build_dataset.py --repro
test:   ; $(PY) -m pytest tests -q
graft:  ; graft build
all: data test repro
