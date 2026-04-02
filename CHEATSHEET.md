# Command Cheatsheet

## Starting Up
```
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml ps
```

## Shutting Down
```
docker compose -f docker/compose.yaml down
```

## Checking Logs
```
docker compose -f docker/compose.yaml logs kafka
docker compose -f docker/compose.yaml logs mlflow
```

## Python Environment
- Mac: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

## Git
```
git add .
git commit -m "your message here"
git push
```

## MLflow Dashboard
http://localhost:5001