MAKEFLAGS = --silent --ignore-errors --no-print-directory
.PHONY: start stop build clean healthcheck imageclean push

build: healthcheck Dockerfile
	docker build . --tag sharkzeeh/face:v1

healthcheck: ./healthcheker.sh
	./healthcheker.sh

start: build
	docker run  --rm -it --name face sharkzeeh/face:v1 /bin/sh -c '/bin/bash; cd PFLD-pytorch'

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