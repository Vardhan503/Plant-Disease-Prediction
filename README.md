# Plant Disease Detection API

A high-performance REST API designed to classify plant diseases from leaf images. This application utilizes a custom Convolutional Neural Network (CNN) trained on the PlantVillage dataset to categorize images into 38 distinct classes. It is built with FastAPI for the backend, PostgreSQL for persistent data storage, and is fully containerized using Docker.

## Features

* **Deep Learning Inference:** Classifies plant leaf images with high accuracy using a custom PyTorch CNN architecture.
* **Real-time Predictions:** Fast response times for image processing and classification.
* **Data Persistence:** Automatically saves prediction results, filenames, and confidence scores to a PostgreSQL database.
* **Containerized Environment:** Fully isolated application and database services using Docker and Docker Compose.
* **Input Validation:** Robust file validation using Pydantic schemas.

## Tech Stack

* **Language:** Python 3.9
* **Framework:** FastAPI
* **Machine Learning:** PyTorch, Torchvision
* **Database:** PostgreSQL, SQLAlchemy (ORM)
* **Containerization:** Docker, Docker Compose
* **Data Handling:** NumPy, Pillow

## Project Structure

```text
PLANTDISEASE/
├── app/
│   ├── config.py          # Environment configuration
│   ├── database.py        # Database connection logic
│   ├── inference.py       # Model loading and prediction logic
│   ├── main.py            # FastAPI application entry point
│   ├── models.py          # SQLAlchemy database models
│   └── schemas.py         # Pydantic data schemas
├── models/
│   ├── classes.json       # List of 38 class names
│   └── plant_disease_custom_cnn.pth  # Trained PyTorch model weights
├── notebooks/
│   └── plantdisease.ipynb # Jupyter notebook used for model training
├── plantvillage_dataset/  # Raw dataset directory
├── .env                   # Environment variables (excluded from version control)
├── docker-compose.yaml    # Container orchestration configuration
├── Dockerfile             # API container build definition
└── requirements.txt       # Python dependencies

```

## Prerequisites

Ensure you have the following installed on your machine:

* Docker Desktop
* Git

## Installation and Setup

Follow these steps to set up and run the application locally.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PLANTDISEASE

```

### 2. Configure Environment Variables

Create a file named `.env` in the root directory. You can copy the following configuration:

```text
DATABASE_HOSTNAME=
DATABASE_PORT=
DATABASE_PASSWORD=
DATABASE_NAME=
DATABASE_USERNAME=
```

### 3. Build and Run with Docker

Start the application and database containers using Docker Compose. This will build the API image and pull the PostgreSQL image.

```bash
docker-compose up -d --build

```

The application will be available at `http://localhost:8000`.

## API Usage

### Health Check

**GET** `/`

Returns a status message to verify the API is running.

```json
{
  "message": "Plant Disease API is running"
}

```

### Predict Disease

**POST** `/predict`

Upload a leaf image to generate a prediction.

* **Header:** `Content-Type: multipart/form-data`
* **Body:** form-data key `file` (File upload)

**Response Example:**

```json
{
  "filename": "tomato_early_blight.jpg",
  "prediction": "Tomato_Early_blight",
  "confidence": 0.9854,
  "db_id": 1
}

```

## Database Access

The PostgreSQL database runs inside a Docker container. To access the data from your local machine (using pgAdmin, DBeaver, or terminal), use the following credentials:

* **Host:** `localhost`
* **Port:** `5433` (Mapped from container port 5432)
* **User:** `postgres`
* **Password:** `root`
* **Database:** `plant`
* **Table:** `"Predictions"` (Note the capitalization and quotes if using SQL queries)

## Model Training

The model was trained using the `notebooks/plantdisease.ipynb` file. The training process involves:

1. **Data Augmentation:** Random affine transformations (rotation, shear, zoom) to improve generalization.
2. **Architecture:** A custom 3-block CNN with Batch Normalization and Dropout.
3. **Optimization:** Adam optimizer with `ReduceLROnPlateau` scheduler.

To reproduce the training, ensure the `plantvillage_dataset` folder is populated with the raw image data.

## Future Improvements

* Deploy to a cloud provider using CI/CD pipelines.
