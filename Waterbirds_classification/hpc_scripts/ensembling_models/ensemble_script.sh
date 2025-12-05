#!/bin/bash -l  
#SBATCH --time=2:30:00 
#SBATCH --array=1-3
#SBATCH --mem-per-cpu=64G
#SBATCH --account=aip-ebrahimi  
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4 

module load python/3.10     

dataset=$1 #cfgs/waterbirds.yaml
alpha=$2
temperature=$3
t_size=$4
ensemble_method=$5 
student_model=$6 
teacher_model=$7 
debiasing_method="$9" 
resnet_version_teacher=${11}
resnet_version_student=${12}
num_nodes=1
erm_teachers=${10}
exp_name=${13}
b_teacher_resnet_version=${14}
END=2
num_workers=4
devices=4

convnextv2_version=tiny
vit_version=base
bert_version=tiny
resnet_version=$resnet_version_student

if [ "$teacher_model" == "resnet" ]
then
    query="$query --teacher_resnet_version=$resnet_version_teacher --resnet_version=$resnet_version_student"
    
    if [ "$exp_name" == "DebiasedTeacherSize" ]
    then
            debiased_teacher_size=5
            exp_name="DebiasedTeacherSize/${exp_name}_$debiased_teacher_size"
            query="$query --exp_name $exp_name --finetuned_teachers_size $debiased_teacher_size"
    else   
        if [ "$exp_name" != "None" ]
        then
            query="$query --exp_name $exp_name" 
            
        fi
    fi
else
    query="$query --bert_version large --teacher_resnet_version=18 --resnet_version=18"
fi

query="ensemble.py -c $dataset --temperature=$temperature --resnet_version=18 --bert_version=$bert_version --vit_version=$vit_version --convnextv2_version=$convnextv2_version  --teacher_model=$teacher_model --model=$student_model --num_workers $num_workers --erm_teachers $erm_teachers --num_nodes=$num_nodes --ensemble_method=$ensemble_method  --accelerator=gpu --devices=$devices --teacher_size=$t_size --alpha=$alpha --interactive_mode False"
 

echo python ${query} --debiased_teachers_method "$debiasing_method" --seed=$SLURM_ARRAY_TASK_ID
srun python ${query} --seed=$SLURM_ARRAY_TASK_ID --debiased_teachers_method "$debiasing_method"