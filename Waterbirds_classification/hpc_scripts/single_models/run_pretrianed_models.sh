#!/bin/bash 

resnet_versions=(50 18)
#datasets=(cfgs/waterbirds.yaml cfgs/celeba.yaml cfgs/multinli.yaml cfgs/civilcomments.yaml) 
#datasets=(cfgs/civilcomments.yaml cfgs/multinli.yaml)  
datasets=(cfgs/waterbirds.yaml) 
balance_erms=(False)
bootstraps=(True False)      # runs both non-bootstrap and bootstrap
bootstrap_frac=0.7
bootstrap_n_sets=1
bootstrap_with_replacement=True

#debiasing_methods=("early-stop disagreement self" "dropout disagreement self")
for dataset_conf in "${datasets[@]}" 
do
    if [ "$dataset_conf" == "cfgs/civilcomments.yaml" ]  || [ "$dataset_conf" == "cfgs/multinli.yaml" ]
    then 
        base_model=bert 
       # versions=(base tiny)
        versions=(base)
    else 
        base_model=resnet
        versions=(50 18) 
        #base_model=convnextv2
        #versions=(base tiny) 
    fi

    for version in "${versions[@]}" 
    do
        for balance_erm in "${balance_erms[@]}" 
        do	
            for bootstrap in "${bootstraps[@]}"
            do
                sbatch hpc_scripts/single_models/single_model_script.sh $base_model $dataset_conf $balance_erm $version $bootstrap $bootstrap_frac 
            done
        done
    done
done
