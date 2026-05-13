# Deployment Notes

## Local deployment
1. Install dependencies.
2. Generate data and train model.
3. Run `uvicorn app.main:app --reload`.

## Docker deployment
- Build: `docker build -t ml-deployment-lab .`
- Run: `docker run -p 8000:8000 ml-deployment-lab`
- Or: `docker compose up --build`

## Environment variables
Use `.env.example` as a template.

## API docs
Swagger UI: `http://localhost:8000/docs`

## Production considerations
- External model registry
- Authentication and authorization
- Centralized logs and metrics
- CI/CD and model versioning
