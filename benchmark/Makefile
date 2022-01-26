.ONESHELL:
POETRY=docker run --rm -it --privileged --gpus all -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/subuid:/etc/subuid:ro -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --network=host -e HOME=/home/argon -u=501:6809 -v /home:/home -v /opt:/opt -v /tmp:/tmp -v /mnt:/mnt -w $$(pwd) --entrypoint poetry ar90n/python:latest

.PHONY: setup benchmark flann annoy

setup:
	cd  ann-benchmarks
	$(POETRY) init --python '>= 3.8, < 3.11' -n
	$(POETRY) add $$(cat requirements.txt | sed 's/=.*//g')

$(wildcard results/fashion-mnist-784-euclidean/10/annoy/*.hdf5): setup
	cd  ann-benchmarks
	$(POETRY) run python install.py --algorithm annoy
	$(POETRY) run python run.py --algorithm annoy --dataset fashion-mnist-784-euclidean

annoy: $(wildcard results/fashion-mnist-784-euclidean/10/annoy/*.hdf5)


$(wildcard results/fashion-mnist-784-euclidean/10/flann/*.hdf5): setup
	cd  ann-benchmarks
	$(POETRY) run python install.py --algorithm flann
	$(POETRY) run python run.py --algorithm flann --dataset fashion-mnist-784-euclidean

flann: $(wildcard results/fashion-mnist-784-euclidean/10/flann/*.hdf5)


benchmark: flann  annoy
	cd  ann-benchmarks
	sudo chmod -R 777 ./results
	$(POETRY) run python plot.py --dataset fashion-mnist-784-euclidean
