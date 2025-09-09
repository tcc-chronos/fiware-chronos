.PHONY: dev-up run format lint

dev-up:
	./scripts/dev_up.sh

run:
	./scripts/run_api.sh

format:
	./scripts/format.sh

lint:
	./scripts/lint.sh
