#!/usr/bin/env bash
devices=$1
docker run -u $(id -u):$(id -g)  --gpus '"device='$devices'"' --ipc=host --rm --volume="/Neuronal/dataset_preprocesado_png/:/Neuronal/dataset_preprocesado_png/" --volume="/Neuronal/bedroom_dataset_preprocesado256/:/Neuronal/bedroom_dataset_preprocesado256/" --volume="/home/claramingyue/:/home/claramingyue/" ddpm-pytorch241-cuda121 bash -c "cd /home/claramingyue/DDPM; python3 DDPM.py"