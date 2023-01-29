.ONESHELL:
POETRY=poetry
GO=go

.PHONY: build setup benchmark annoy-siftsmall linear-siftsmall countrymaam-siftsmall plot

build:
	$(GO) build -o bin/countrymaam ../../cmd/countrymaam/main.go

setup: build
	cd annbench
	cp ../countrymaam.py ./annbench/algo/countrymaam.py
	cp ../countrymaam.yaml ./conf/algo/countrymaam.yaml
	patch -p 1 -forward < ../proxy.patch
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

countrymaam-siftsmall: setup
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=countrymaam


benchmark: annoy-siftsmall linear-siftsmall countrymaam-siftsmall
	cd annbench
	$(POETRY) run python plot.py

plot: 
	cd annbench
	$(POETRY) run python plot.py
