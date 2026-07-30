.PHONY: help install test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

test:  ## Run the test suite
	uv run pytest
