install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
test: 
	python -m pytest -vv --cov=app test_app.py	

format: 
	black *.py

lint: 
	pylint --disable=R,C, *.py

model:
	python3 download_model.py

all: install lint test format model