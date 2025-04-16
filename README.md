# Cricket Analysis API

This project provides a REST API for cricket match predictions, player ratings, team performance analysis, and best 11 selection.

## Features

- Match Prediction API
- Player Rating API
- Team Performance API
- Best 11 Selection API

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The API will be available at http://localhost:5000

## API Endpoints

### 1. Match Prediction
```
POST /api/predict/match
{
    "team1": "India",
    "team2": "Australia",
    "ground": "MCG"
}
```

### 2. Player Ratings
```
GET /api/player/ratings/<player_id>
```

### 3. Team Performance
```
GET /api/team/performance/<team_name>
```

### 4. Best 11
```
GET /api/team/best11/<team_name>
```

## Production Deployment

1. Build the Docker image:
```bash
docker build -t cricket-analysis-api .
```

2. Run the container:
```bash
docker run -d -p 5000:5000 cricket-analysis-api
```

## Environment Variables

- FLASK_ENV: development/production
- FLASK_APP: app.py

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 