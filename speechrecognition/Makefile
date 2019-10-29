CUR_DIR = $(CURDIR)

all: build_docker create_dataset

build_docker:
	docker build . -t speechrecognition

create_dataset:
	docker run -it -v $(CURDIR):/speechrecognition/ speechrecognition python3 src/dataset/create_dataset.py
