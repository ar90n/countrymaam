.ONESHELL:
POETRY=poetry
GO=go

.PHONY: build setup benchmark annoy-siftsmall linear-siftsmall countrymaam-siftsmall plot

build:
	$(GO) build -o lib/libcountrymaam_wrapper.so -buildmode=c-shared ../wrapper/countrymaam_wrapper.go


setup: build
	cd annbench
	cp ../countrymaam.py ./annbench/algo/countrymaam.py
	cp ../../wrapper/countrymaam_wrapper.py ./annbench/algo/countrymaam_wrapper.py
	cp ../lib/libcountrymaam_wrapper.so ./annbench/algo/libcountrymaam_wrapper.so
	cp ../countrymaam.yaml ./conf/algo/countrymaam.yaml
	patch -p 1 -forward < ../proxy.patch
	$(POETRY) init --python '>= 3.8, < 3.11' -n
	$(POETRY) add $$(cat requirements.txt | sed 's/=.*//g')
	$(POETRY) add faiss-cpu
	$(POETRY) install

annbench/dataset/siftsmall:
	cd annbench
	$(POETRY) run python download.py dataset=siftsmall

annbench/dataset/sift1m:
	cd annbench
	$(POETRY) run python download.py dataset=sift1m


annoy-siftsmall: setup annbench/dataset/siftsmall
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=annoy

linear-siftsmall: setup annbench/dataset/siftsmall
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=linear

countrymaam-siftsmall: setup annbench/dataset/siftsmall
	cd annbench
	$(POETRY) run python run.py dataset=siftsmall algo=countrymaam

annoy-sift1m: setup annbench/dataset/sift1m
	cd annbench
	$(POETRY) run python run.py dataset=sift1m algo=annoy

linear-sift1m: setup annbench/dataset/sift1m
	cd annbench
	$(POETRY) run python run.py dataset=sift1m algo=linear

countrymaam-sift1m: setup annbench/dataset/sift1m
	cd annbench
	$(POETRY) run python run.py dataset=sift1m algo=countrymaam

benchmark-siftsmall: annoy-siftsmall linear-siftsmall countrymaam-siftsmall
	cd annbench
	$(POETRY) run python plot.py


benchmark-sift1m: annoy-sift1m linear-sift1m countrymaam-sift1m
	cd annbench
	$(POETRY) run python plot.py

plot: 
	cd annbench
	$(POETRY) run python plot.py
