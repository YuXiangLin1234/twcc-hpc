#! /bin/bash
#SBATCH --job-name=llama-ft           
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1                     
#SBATCH --cpus-per-task=8                        #  number of cores per tasks
#SBATCH --gres=gpu:8                             #  number of gpus
#SBATCH --output=%x-%j.out                       #  output file name
#SBATCH --error=%x-%j.error                      #  error file name 
#SBATCH --account=MST112330
#SBATCH --partition=gp1d
#SBATCH --time=0-10:00:00                                               

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# init environment
# https://hackmd.io/@kmo/twcc_hpc_conda
module purge
module load miniconda3
module load cuda/11.7
module load gcc9/9.3.1

# TODO env path
conda activate /work/yuxiang1234/miniconda3/envs/llm-train
export PYTHONUSERBASE=$CONDA_PREFIX
export HF_H0ME=/work/yuxiang1234/hf_home
export TRANSFORMERS_CACHE=/work/yuxiang1234/cache

# TODO: arguments
srun --jobid $SLURM_JOBID bash -c "torchrun --nnodes 4 --nproc_per_node 8 \
  --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:2234 \
   llama-recipes/examples/finetuning.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --output_dir /work/yuxiang1234/LLM/model-ckpt \
  --save_metrics True \
  --batch_size_training 4 \
  --gradient_accumulation_steps 1 \
  --enable_fsdp \
  --peft_method None \
  --use_peft False \
  --freeze_layers False \
  --quantization False \
  --num_epochs 2 \
  --dataset samsum_dataset \
  --use_fp16 \
  --lr 5e-6"

echo "Finish $(date)"
