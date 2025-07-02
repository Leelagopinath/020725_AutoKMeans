.PHONY: run test lint preprocess docker-build docker-run

run:
	streamlit run src/application/app.py

test:
	pytest -v tests/

lint:
	flake8 src/
	black --check src/

preprocess:
	bash scripts/preprocess.sh data/raw/input_data.csv

docker-build:
	docker build -t kmeans-automation .

docker-run:
	docker run -p 8501:8501 -v $(pwd)/data:/app/data kmeans-automation