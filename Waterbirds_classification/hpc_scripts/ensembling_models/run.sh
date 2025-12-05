#!/bin/bash 

#datasets=(compas employment employment_race new_adult mobility poverty pub_coverage)
#datasets=(ColoredMNIST-Skewed0.005-Severity4 ColoredMNIST-Skewed0.01-Severity2)
#sensitive_attr=Color
#target_attr=Digit
#datasets=(CelebA)
#sensitive_attr=Male
#target_attr=Blond_Hair
datasets=(cfgs/multinli.yaml cfgs/waterbirds.yaml cfgs/celeba.yaml cfgs/civilcomments.yaml)
datasets=(cfgs/waterbirds.yaml)
alphas=(1.0)
teacher_sizes=(10) 
temperatures=(4)
b_teacher_resnet_version=50

erm_teacherss=(False True)
resnet_version_students=(18)
resnet_version_teachers=(50)
ensemble_methods=(ModelSoup ENS) #  ENS  AVERAGE_LOSS RANDOM AEKD OURS3
exp_name="None"  
exp_name="StudentCapacityR" 

#Cs=(0.1 0.3 0.6 1.0) 
Cs=(1.0) 
#betas=(0.001 0.002 0.02 0.1)
betas=(0.02) 
debiasing_methods=("early-stop disagreement self" "group-unbalanced retraining" "group-balanced retraining" "early-stop misclassification self" "dropout disagreement self")
debiasing_methods=("group-balanced retraining")

for erm_teachers in "${erm_teacherss[@]}" 
    do 
    for dataset in "${datasets[@]}" 
        do 
        if [ "$dataset" == "cfgs/civilcomments.yaml" ]  || [ "$dataset" == "cfgs/multinli.yaml" ]
        then
            student_models=(bert)
            student_model=bert
            teacher_models=(bert) 
        else
            student_models=(resnet)
            student_model=resnet
            teacher_models=(resnet) 
        fi
        for debiasing_method in "${debiasing_methods[@]}"
            do 
            for beta in "${betas[@]}"
            do 
                for teacher_model in "${teacher_models[@]}" 
                do
                    for t_size in "${teacher_sizes[@]}"
                    do
                        for alpha in "${alphas[@]}" 
                        do 
                            for ensemble_method in "${ensemble_methods[@]}"
                            do
                                for temperature in "${temperatures[@]}" 
                                    do
                                        if [ "$t_size" == 1 ] && [ "$ensemble_method" == "AEKD" ]
                                        then
                                            echo "Teacher size $t_size $ensemble_method wrong"
                                            #echo "OK $t_size $ensemble_method"
                                        else 
                                            if [ "$teacher_model" == "resnet" ]
                                            then
                                                for resnet_version_teacher in "${resnet_version_teachers[@]}" 
                                                do
                                                    for resnet_version_student in "${resnet_version_students[@]}" 
                                                    do
                                                        sbatch /scratch/o/omata/bias-eval-model-soup/hpc_scripts/ensembling_models/ensemble_script.sh $dataset $alpha $temperature $t_size $ensemble_method $student_model $teacher_model $beta "$debiasing_method" $erm_teachers $resnet_version_teacher $resnet_version_student $exp_name $b_teacher_resnet_version
                                                    done
                                                done
                                            else 
                                                sbatch /scratch/o/omata/bias-eval-model-soup/hpc_scripts/ensembling_models/ensemble_script.sh $dataset $alpha $temperature $t_size $ensemble_method $student_model $teacher_model $beta "$debiasing_method" $erm_teachers                             
                                            fi
                                        fi
                                    done 
                            done 
                        done 
                    done 
                done 
            done  
        done  
    done  
done  