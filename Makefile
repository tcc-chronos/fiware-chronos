.PHONY: up stop run format lint

up:
	./scripts/docker/docker_up.sh $(ARGS)

stop:
	./scripts/docker/docker_stop.sh

run:
	./scripts/run_api.sh

format:
	./scripts/format.sh

lint:
	./scripts/lint.sh
