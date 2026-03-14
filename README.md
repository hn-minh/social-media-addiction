# Social Media Addiction Predictor

**Web Demo:** [https://social-ai-predictor.me]

**API Documentation:** [https://api.social-ai-predictor.me/docs]

An end-to-end, cloud-native machine learning system designed to predict user social media addiction levels based on behavioral data. The project features a fully stateless architecture, real-time data collection, and an automated CI/CD pipeline deployed on AWS.

## Key Features

* **Stateless & Cloud-Native:** Model artifacts are versioned and hosted on DagsHub (MLflow), while user interaction data is stored in a Cloud PostgreSQL database.
* **Automated CI/CD:** Integrated GitHub Actions pipeline featuring automated unit/integration testing (via Pytest with Mocking) and seamless deployment to AWS EC2.
* **Lightning-Fast Builds:** Containerized using Docker and Docker Compose.
* **Production-Ready Security:** Hosted behind an Nginx reverse proxy with automated Let's Encrypt SSL certificates (HTTPS) and strict CORS policies.

## Tech Stack

* **Machine Learning & Tracking:** Scikit-learn, Pandas, MLflow, DagsHub
* **Backend API:** FastAPI, SQLAlchemy, Pydantic, Pytest
* **Frontend:** Streamlit
* **Database:** PostgreSQL (Cloud DBaaS)
* **DevOps & Infrastructure:** Docker, Docker Compose, GitHub Actions, AWS EC2, Nginx, Let's Encrypt, `uv`

## Deployment Architecture

This application is deployed on an **AWS EC2 (Ubuntu)** instance following a strict CI/CD lifecycle.

The pipeline triggers automatically on every push to the `main` branch:

1. Provisions an isolated GitHub Actions environment.
2. Runs robust Mock Tests using Pytest to ensure logic integrity.
3. Upon success, authenticates via SSH to the AWS EC2 instance.
4. Rebuilds and restarts the Docker containers automatically with zero downtime.
5. All traffic is routed securely via Nginx with enforced HTTPS.