install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	
lint:
	flake8 *.py

test:
	python -m pytest -vv --cov=analysis *.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

run:
	python analysis.py

all: install format lint test