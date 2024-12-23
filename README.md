# FastAPI Sentiment Analysis API

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

## Overview
This project is a **Sentiment Analysis API** built using **FastAPI**. The API utilizes a pre-trained machine learning model to predict the sentiment (Positive or Negative) of text data. It is fully containerized with Docker, making it easy to deploy locally or on any cloud platform.

---

## Features
- **Sentiment Analysis**: Predicts whether a given text is positive or negative.
- **FastAPI**: High-performance web framework for serving the model.
- **Docker**: Simplifies deployment and environment setup.
---

## Installation

### Prerequisites
- Python 3.8 or higher
- Docker & Docker Compose



### Run Locally Without Docker
**1.To set up a virtual environment and install dependencies:**
### Setup Virtual Environment 

- python3 -m venv env
- source env/bin/activate  # On Windows: env\Scripts\activate
- pip install -r requirements.txt

**2. Train the Model**

- python src/models/train_model.py

**3. Start the API**

- uvicorn main:app --reload

**4. Access the API**

Once the server is running, open your browser or API testing tool and navigate to:

Swagger UI: http://127.0.0.1:8000/docs

### Running with Docker
**Build the Docker Image**

```bash
docker build -t fastapi-sentiment .
#Run the Container
docker run -p 8000:8000 fastapi-sentiment





