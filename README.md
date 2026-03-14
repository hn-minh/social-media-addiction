# Social Media Addiction Predictor

An end-to-end, cloud-native machine learning system designed to predict user social media addiction levels based on behavioral data. The project features a fully stateless architecture, real-time data collection, and an automated CI/CD pipeline deployed on AWS.

## 🚀 Key Features

* **Stateless & Cloud-Native:** Model artifacts are versioned and hosted on DagsHub (MLflow), while user interaction data is stored in a Cloud PostgreSQL database.
* **Automated CI/CD:** Integrated GitHub Actions pipeline featuring automated unit/integration testing (via Pytest with Mocking) and seamless deployment to AWS EC2.
* **Lightning-Fast Builds:** Containerized using Docker and Docker Compose.
* **Production-Ready Security:** Hosted behind an Nginx reverse proxy with automated Let's Encrypt SSL certificates (HTTPS) and strict CORS policies.

## 🛠️ Tech Stack

* **Machine Learning & MLOps:** Scikit-learn, Pandas, MLflow, DagsHub
* **Backend API:** FastAPI, SQLAlchemy, Pydantic, Pytest
* **Frontend:** Streamlit
* **Database:** PostgreSQL
* **DevOps & Infrastructure:** Docker, Docker Compose, GitHub Actions, AWS EC2, Nginx, Let's Encrypt, `uv`

## ⚙️ Local Development Setup

### Prerequisites

* [Docker](https://www.docker.com/) and Docker Compose installed.
* A Cloud PostgreSQL database URL.
* A DagsHub account with MLflow Tracking credentials.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/social-media-addiction.mlflow.git
cd social-media-addiction.mlflow

```


2. **Configure Environment Variables:**
Create a `.env` file in the root directory and add your credentials:
```env
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
CLOUD_DB_URL=postgresql://user:password@host:5432/dbname
FRONTEND_URL=http://localhost:8501

```


3. **Run with Docker Compose:**
```bash
docker compose up --build

```


4. **Access the application:**
* Web Interface: `http://localhost:8501`
* API Documentation (Swagger UI): `http://localhost:8000/docs`



## ☁️ Deployment Architecture

This application is deployed on an **AWS EC2 (Ubuntu)** instance.
The CI/CD pipeline triggers on every push to the `main` branch:

1. Provisions an isolated Ubuntu environment.
2. Runs robust Mock Tests using Pytest.
3. Upon success, SSH connects to the EC2 instance to pull the latest code.
4. Rebuilds and restarts the Docker containers automatically.
5. Traffic is routed securely via Nginx and HTTPS.