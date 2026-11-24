.PHONY: up down logs test clean

# Start the whole stack: backend :8000, frontend :3000
up:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - add your real keys to LLM_API_KEYS")
	docker compose up --build -d
	@echo "Frontend: http://localhost:3000   Backend: http://localhost:8000/health"

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && pip install -q -r requirements-dev.txt && python -m pytest -q

clean:
	docker compose down -v
