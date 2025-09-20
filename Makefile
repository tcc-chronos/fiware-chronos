.PHONY: up build stop run format lint

up:
	./scripts/docker/docker_up.sh $(ARGS)

build:
	./scripts/docker/docker_build.sh $(ARGS)

stop:
	./scripts/docker/docker_stop.sh

run:
	./scripts/run_api.sh

format:
	./scripts/format.sh

lint:
	./scripts/lint.sh
