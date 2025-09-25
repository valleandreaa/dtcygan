# dtcygan

## Python Setup
- Target Python version: 3.12.
- Install dependencies with: pip install -r requirements.txt

## Docker Usage
1. Build the image: docker compose build
2. Start an interactive shell: docker compose run --rm app
3. Run project commands inside the container from /workspace
4. Stop any lingering containers: docker compose down (if you started services in detached mode)

The provided Dockerfile uses the official python:3.12-slim base image and installs the packages listed in requirements.txt.
