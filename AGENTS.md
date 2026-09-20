# Document Understanding

Exploration of document understanding and information retrieval techniques.

## Development

- Python language
- uv dependency manager
- docker compose for running services
- SOLID principles
- Concise comments (never inline)
- Google style docstrings (always state args and returns when not 'None')
- Ruff linter
- Never use functions inside functions (or classes inside classes)
- Always use imports in the top of files (never imports exclusive for functions)
- Explicit arguments names in functions (avoid positionals)
- Layered architecture (folders: `src/services/`, `src/core/`, `src/schemas/`)
- DTO for more than one object returned in a method (pydantic)
- Services configurations in JSON (`configs/` directory)
- Every service must be in the README.md file, containing:
  - Inputs: config payload as table (key, description, default value)
  - Usage: docker compose run command
  - Outputs: generated artifacts and metrics (tree structure and/or table)
- Every service should run as a docker compose service
- `data/` directory contains the dataset to be used
- `checkpoints/` directory contains the AI models
- `tests/` directory contains unit tests
- Working with reduced VRAM (4 GB)
