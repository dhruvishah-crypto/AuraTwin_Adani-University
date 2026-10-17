.PHONY: help install generate test train run clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install required Python dependencies"
	@echo "  make generate  - Generate synthetic patient EHR data"
	@echo "  make test      - Run pytest test suite"
	@echo "  make train     - Train machine learning models"
	@echo "  make run       - Launch Streamlit dashboard"
	@echo "  make all       - Run complete pipeline (generate -> test -> run)"

install:
	pip install -r requirements.txt

generate:
	python src/generate_data.py --num_patients 300 --seed 42 --output_dir data

test:
	pytest -v

train:
	python src/train_model.py

run:
	streamlit run app.py

all: generate test run
