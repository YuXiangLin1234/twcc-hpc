# Example usage of TWCC HPC

## Install Env
!!! Check your pip path by `which pip` in the conda env!!!
```
cd llama-recipes
pip install -r requirements.txt
pip install -e .
```

## Execute
Change TODOs in `llama-ft.slurm` and then execute.

```
sbatch llama-ft.slurm
```