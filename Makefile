MAKEFLAGS = --silent --ignore-errors --no-print-directory
.PHONY: start stop build clean healthcheck imageclean push ABC

healthcheck: ./healthchecker.sh
	./healthchecker.sh

build: healthcheck Dockerfile
	docker build . --tag sharkzeeh/face:v1

start: build
	xhost +
	docker run -it --name face --net=host --env="DISPLAY" sharkzeeh/face:v1 /bin/bash

stop:
	docker stop face

contclean: stop
	docker rm -f face

imageclean: stop contclean
	docker rmi -f `docker images -a | grep face | grep -Po '\b\w{12}\b'`
	docker rmi `docker images --filter "dangling=true" -q --no-trunc` >/dev/null 2>&1
	docker volume rm -f $(docker volume ls -f "dangling=true")

push: build
	docker tag sharkzeeh/face:v1  sharkzeeh/face:v1
	docker push sharkzeeh/face:v1