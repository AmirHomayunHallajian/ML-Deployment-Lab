install:
	pip install -r requirements.txt

data:
	python scripts/generate_synthetic_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

test:
	pytest

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t ml-deployment-lab .

docker-run:
	docker run -p 8000:8000 ml-deployment-lab

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
