.ONESHELL:
POETRY=poetry
PARALLELISM=1

.PHONY: setup benchmark flann annoy countrymaam-kd countrymaam-rp

setup:
	cd ann-benchmarks
	cp ../countrymaam.py ./ann_benchmarks/algorithms
	cp ../Dockerfile.countrymaam ./install
	patch -N < ../patch_algos.yaml
	$(POETRY) init --python '>= 3.8, < 3.11' -n
	$(POETRY) add $$(cat requirements.txt | sed 's/=.*//g')

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
