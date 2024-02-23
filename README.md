# Example usage of TWCC HPC

## Install Env
!!! Check your pip path by `which pip` in the conda env!!!
```
pip install -r llama-recipes/requirements.txt
```

## Execute
Change TODOs in `llama-ft.slurm` and then execute.

```
sbatch llama-ft.slurm
```

## TODO
The pretraining code of ASBC is placed in `llama-recipes/src/llama_recipes/datasets/alpaca_dataset.py`, but there may be some bugs making that training loss won't go down.
