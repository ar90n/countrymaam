.ONESHELL:
POETRY=poetry
GO=go
DOCKER=docker
PARALLELISM=1
pwd=$(pwd)

.PHONY: build setup benchmark flann annoy countrymaam-kd countrymaam-rp test

build:
	$(DOCKER) run --rm -v $(PWD)/../../:$(PWD)/../../ -w $(PWD) golang go  build -o lib/libcountrymaam_wrapper.so -buildmode=c-shared ../wrapper/countrymaam_wrapper.go

setup: build
	cd ann-benchmarks
	cp ../countrymaam.py ./ann_benchmarks/algorithms
	cp ../../wrapper/countrymaam_wrapper.py ./ann_benchmarks/algorithms
	cp ../lib/libcountrymaam_wrapper.so ./ann_benchmarks/algorithms
	cp ../Dockerfile.countrymaam ./install/Dockerfile.countrymaam-kd
	patch -N < ../patch_algos.yaml
	patch -p1 -N < ../patch_runner.py
	$(POETRY) init --python '>= 3.8, < 3.11' -n
	$(POETRY) add $$(cat requirements.txt | sed 's/=.*//g')
	$(POETRY) add docker

annoy: setup
	cd ann-benchmarks
	$(POETRY) run python install.py --algorithm annoy
	$(POETRY) run python run.py --algorithm annoy --dataset fashion-mnist-784-euclidean --parallelism $(PARALLELISM); true

flann: setup
	cd ann-benchmarks
	$(POETRY) run python install.py --algorithm flann
	$(POETRY) run python run.py --algorithm flann --dataset fashion-mnist-784-euclidean --parallelism $(PARALLELISM); true

countrymaam-kd: setup
	cd ann-benchmarks
	$(POETRY) run python install.py --algorithm countrymaam-kd
	$(POETRY) run python run.py --algorithm countrymaam-kd --dataset fashion-mnist-784-euclidean --parallelism $(PARALLELISM); true

countrymaam-rp: setup
	cd ann-benchmarks
	$(POETRY) run python install.py --algorithm countrymaam-rp
	$(POETRY) run python run.py --algorithm countrymaam-rp --dataset fashion-mnist-784-euclidean --parallelism $(PARALLELISM); true

benchmark: annoy flann countrymaam-kd countrymaam-rp
	cd ann-benchmarks
	$(POETRY) run python plot.py --dataset fashion-mnist-784-euclidean -Y log -X a4
