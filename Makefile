.PHONY: check lint type test crap mutate docs run

check: lint type test crap  ## Everything CI would run, fast

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run ty check src tools

test:
	uv run pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=json:coverage.json --cov-fail-under=100

crap: test
	uv run python -m tools.crap coverage.json 6

mutate:
	uv run mutmut run --max-children 4
	uv run mutmut results

docs:
	uv run mkdocs build --strict

run:  ## Download, build, write the card and publish to the Hub
	uv run georeset run --publish
