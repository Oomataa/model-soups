#!/bin/bash -l 
#SBATCH --time=6:00:00  
#SBATCH --array=1-50  
#SBATCH --mem-per-cpu=64G
#SBATCH --account=aip-ebrahimi  
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4 
#SBATCH --ntasks-per-node=1

module load httpproxy/1.0
module load python/3.10 

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

#datatset=cfgs/waterbirds.yaml 
datatset=$2
num_nodes=1 
base_model=$1
balance_erm=$3
version=$4
bootstrap=${5:-False}
bootstrap_frac=${6:-0.7}
bootstrap_n_sets=${7:-50}
bootstrap_with_replacement=${8:-True}
seed=$SLURM_ARRAY_TASK_ID
retrain=True

query="python single_model.py -c $datatset --seed=$seed --balance_erm $balance_erm --model=$base_model --num_nodes=$num_nodes --interactive_mode False --retrain $retrain  --bootstrap $bootstrap --bootstrap_frac $bootstrap_frac --bootstrap_n_sets $bootstrap_n_sets --bootstrap_with_replacement $bootstrap_with_replacement"

if [ "$base_model" == "resnet" ]
then
    query="$query --resnet_version=$version"
elif [ "$base_model" == "bert" ]; then
    query="$query --bert_version=$version"
elif [ "$base_model" == "convnextv2" ]; then
    query="$query --convnextv2_version=$version"
fi

echo ${query}
${query}
