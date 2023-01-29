.ONESHELL:
POETRY=docker run --rm -it --privileged --gpus all -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/subuid:/etc/subuid:ro -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --network=host -e HOME=/home/argon -u=501:6809 -v /home:/home -v /opt:/opt -v /tmp:/tmp -v /mnt:/mnt -w $$(pwd) --entrypoint poetry ar90n/python:latest

.PHONY: setup benchmark flann annoy

setup:
	cd annbench
	$(POETRY) init --python '>= 3.8, < 3.11' -n
	$(POETRY) add $$(cat requirements.txt | sed 's/=.*//g')
	$(POETRY) add faiss-cpu
	$(POETRY) run python download.py dataset=siftsmall

annoy-siftsmall: setup
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=annoy

linear-siftsmall: setup
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=linear

benchmark: annoy-siftsmall linear-siftsmall
	cd annbench
	$(POETRY) run python plot.py
