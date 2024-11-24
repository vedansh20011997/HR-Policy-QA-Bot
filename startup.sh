#!/bin/bash

docker build -t qa-bot .
docker run -d -p 8000:8000 qa-bot