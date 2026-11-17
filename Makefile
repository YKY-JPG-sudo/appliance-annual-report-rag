.PHONY: setup download build run test evaluate

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

download:
	.venv/bin/python scripts/download_reports.py

build:
	.venv/bin/python scripts/build_index.py

run:
	.venv/bin/streamlit run app.py --server.headless true

test:
	.venv/bin/python -m unittest discover -v

evaluate:
	.venv/bin/python scripts/evaluate.py
