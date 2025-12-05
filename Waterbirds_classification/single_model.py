"""Main file for last-layer retraining experimentation."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from copy import deepcopy
from distutils.util import strtobool
import os
import os.path as osp
import pickle
import sys
import pandas as pd
# Imports Python packages.
from configargparse import Parser
import numpy as np
import torch
# Imports PyTorch packages.
from pytorch_lightning import Trainer

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.civilcomments import CivilComments
from milkshake.datamodules.disagreement import Disagreement
from milkshake.datamodules.multinli import MultiNLI
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.datamodules.mnist import ColoredMNIST, MnistDataset
from milkshake.imports import valid_models_and_datamodules
from milkshake.main import load_weights, main
from milkshake.utils import get_weights
from utils import save_results

class WaterbirdsDisagreement(Waterbirds, Disagreement):
    """DataModule for the WaterbirdsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CelebADisagreement(CelebA, Disagreement):
    """DataModule for the CelebADisagreeement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CivilCommentsDisagreement(CivilComments, Disagreement):
    """DataModule for the CivilCommentsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
class MultiNLIDisagreement(MultiNLI, Disagreement):
    """DataModule for the MultiNLIDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class ColoredMNISTDisagreement(ColoredMNIST, Disagreement):
    """DataModule for the MultiNLIDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs) 

def set_erm_training_parameters(args):
    if args.datamodule == "waterbirds":
        args.datamodule_class = Waterbirds
        args.num_classes = 2 
    elif args.datamodule == "celeba":
        args.datamodule_class = CelebA
        args.num_classes = 2 
    elif args.datamodule == "civilcomments":
        args.datamodule_class = CivilComments
        args.num_classes = 2 
    elif args.datamodule == "multinli":
        args.datamodule_class = MultiNLI
        args.num_classes = 3 
    else:
        raise ValueError(f"DataModule {args.datamodule} not supported.")

def set_training_parameters(args):
    if args.datamodule == "waterbirds":
        args.datamodule_class = WaterbirdsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_steps = 250
        args.finetune_lrs = [1e-4, 1e-3, 1e-2]
    elif args.datamodule == "mnist":
        #MnistDataset(root="data").download()
        args.datamodule_class = ColoredMNISTDisagreement
        args.num_classes = 10
        args.retrain_epochs = 100
        args.finetune_steps = 250
        args.finetune_lrs = [1e-4, 1e-3, 1e-2]
    elif args.datamodule == "celeba":
        args.datamodule_class = CelebADisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_steps = 250
        args.finetune_lrs = [1e-4, 1e-3, 1e-2]
    elif args.datamodule == "civilcomments":
        args.datamodule_class = CivilCommentsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 10
        args.finetune_steps = 500
        args.finetune_lrs = [1e-6, 1e-5, 1e-4]
    elif args.datamodule == "multinli":
        args.datamodule_class = MultiNLIDisagreement
        args.num_classes = 3
        args.retrain_epochs = 10
        args.finetune_steps = 500
        args.finetune_lrs = [1e-6, 1e-5, 1e-4]
    else:
        raise ValueError(f"DataModule {args.datamodule} not supported.")

    args.finetune_num_datas = [20, 100, 500]
    args.dropout_probs = [0.5, 0.7, 0.9]
    args.early_stop_nums = [1, 2, 5]

def load_erm():
    if osp.isfile("teacher_erm.pkl"):
        with open("teacher_erm.pkl", "rb") as f:
            erm = pickle.load(f)
    else: 
        datasets = ["waterbirds", "celeba", "civilcomments", "multinli", "mnist"]
        seeds = range(1, 51) 
        resnet_versions = [18, 34, 50, 101, 152] 
        bert_versions = ["tiny", "mini", "small", "medium", "base", "large"]
        resnet_models = [f'resnet_{v}' for v in resnet_versions]
        bert_models = [f'bert_{v}' for v in bert_versions]
        conv_versions = ["atto", "femto", "pico", "nano",
                        "tiny", "base", "large", "huge"]
        vit_versions = ["base", "large"]
        conv_models = [f'convnextv2_{surfix}' for surfix in conv_versions]
        base_models = resnet_models + conv_models + bert_models + [f'vit_{v}' for v in vit_versions]
        erm = {}
        debiased_status = [True, False]
        balance_erms = [True, False]
        bootstrap_opts = [True, False]
        for d in datasets:
            erm[d] = {}
            for s in seeds:
                erm[d][s] = {}
                for model in base_models:
                    erm[d][s][model] = {}
                    for is_balance_erm in balance_erms:
                        erm[d][s][model][is_balance_erm] = {}
                        for is_debiased in debiased_status:
                            erm[d][s][model][is_balance_erm][is_debiased] = {}
                            for is_bootstrap in bootstrap_opts:
                                erm[d][s][model][is_balance_erm][is_debiased][is_bootstrap] = {"version": -1, "metrics": []}
        with open("teacher_erm.pkl", "wb") as f:
            pickle.dump(erm, f)

    return erm

def dump_erm(args, curr_erm, debiased=False):
     # Try to write the file with retries if it's being used by another process
    max_retries = 5
    retry_delay = 1  # seconds
    for attempt in range(max_retries):
        try:
            erm = load_erm()
            model = get_model_name(args)
                
            erm[args.datamodule][args.seed][model][args.balance_erm][debiased][args.bootstrap] = curr_erm
    
            with open("teacher_erm.pkl", "wb") as f:
                pickle.dump(erm, f)
            break  # Successfully wrote the file, exit the loop
        except IOError as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"File is being used by another process. Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to write file after {max_retries} attempts: {e}")
                #raise  # Re-raise the exception if all retries failed

def get_current_erm(debiased=False):
    erm = load_erm()
    model_name = model = get_model_name(args)
    return erm[args.datamodule][args.seed][model_name][args.balance_erm][debiased][args.bootstrap]

def reset_fc_hook(model):
    model.reset_fc_hook()

def get_pretrained_erm_teacher_weights(args, resnet_pretrained=True):
    if osp.isfile("teacher_erm.pkl"):
        with open("teacher_erm.pkl", "rb") as f:
            erm = pickle.load(f)
    else:
        raise FileNotFoundError("Pretrained teachers models not founds")

    t = erm[args.datamodule][args.seed][args.resnet_version][resnet_pretrained]
    if t["version"] != -1:
        weights = get_weights(args, t["version"], idx=-1) 
        return weights
    else:
        raise FileNotFoundError(f"Skept teacher ResnetNet{args.resnet_version}_{args.seed} version {t['version']} not found")

def print_metrics(metrics, log_dir=None):
    val_metrics, test_metrics = metrics

    val_group_accs = [
        acc for name, acc in val_metrics[0].items() if "group" in name
    ]
    val_avg_acc = val_metrics[0]["val_acc"]
    val_avg_acc = round(val_avg_acc * 100, 1)
    val_worst_group_acc = min(val_group_accs) 
    val_worst_group_acc = round(val_worst_group_acc * 100, 1)
    print(f"Val Average Acc: {val_avg_acc}")
    print(f"Val Worst Group Acc: {val_worst_group_acc}")

    if test_metrics:
        test_group_accs = [
            acc for name, acc in test_metrics[0].items() if "group" in name
        ]
        test_avg_acc = test_metrics[0]["test_acc"]
        test_avg_acc = round(test_avg_acc * 100, 1)
        test_worst_group_acc = min(test_group_accs) 
        test_worst_group_acc = round(test_worst_group_acc * 100, 1)
        print(f"Test Average Acc: {test_avg_acc}")
        print(f"Test Worst Group Acc: {test_worst_group_acc}")
    if log_dir:
        save_results(metrics, args.seed, log_dir)
        #save_results(metrics, result_dir=log_dir)
    print()
 
def get_model_name(args):
    model_name = args.model
    if args.model=='bert':
        model_name+=f"_{args.bert_version}"
    elif args.model=='resnet':
        model_name+=f"_{args.resnet_version}"
    elif args.model=='convnextv2':
        model_name+=f"_{args.convnextv2_version}"
    elif args.model=='vit':
        model_name+=f"_{args.vit_version}" 
    return model_name

def print_results(erm_metrics, results, keys):
    print("---Experiment Results---")
    print("\nERM")
    print_metrics(erm_metrics)
    #save_results(erm_metrics, result_dir="ERM")

    for key in keys:
        print(key.title())
        if "self" in key:
            print(f"Best params: {results[key]['params']}")
        print_metrics(results[key]["metrics"])

        #save_results(results[key]["metrics"], result_dir=key)
    

def finetune_last_layer(
    args,
    finetune_type,
    model_class,
    dropout_prob=0,
    early_stop_weights=None,
    finetune_num_data=None,
    worst_group_pct=None,
):
    disagreement_args = deepcopy(args)
    disagreement_args.finetune_type = finetune_type
    disagreement_args.dropout_prob = dropout_prob

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.lr_scheduler = "step"
    finetune_args.lr_steps = []
    finetune_args.lr = args.finetune_lr

    # Sets parameters for finetuning (first) or retraining (second).
    if finetune_num_data:
        finetune_args.max_epochs = None
        finetune_args.max_steps = args.finetune_steps
        reset_fc = False
    else:
        finetune_args.max_epochs = args.retrain_epochs
        finetune_args.max_steps = -1
        reset_fc = True

    # Don't save the model (we save manually if it is the best).
    #finetune_args.val_check_interval = args.finetune_steps + 1
    #finetune_args.ckpt_every_n_steps = args.finetune_steps + 1

    model = model_class(disagreement_args)
    load_weights(disagreement_args, model)

    early_stop_model = None
    if early_stop_weights:
        early_args = deepcopy(disagreement_args)
        early_args.weights = early_stop_weights
        early_stop_model = model_class(early_args)
        load_weights(early_args, early_stop_model)

    datamodule = args.datamodule_class(
        disagreement_args,
        early_stop_model=early_stop_model,
        model=model,
        num_data=finetune_num_data,
        worst_group_pct=worst_group_pct,
    )

    model_hooks = [reset_fc_hook] if reset_fc else None
    model, val_metrics, test_metrics = main(
        finetune_args,
        model_class,
        datamodule,
        model_hooks=model_hooks,
    )

    return model, val_metrics, test_metrics

def experiment(args, model_class):
    os.makedirs("out", exist_ok=True)









    # Loads ERM paths and metrics from pickle file.
    #erm = load_erm()
   

    
    # Adds experiment-specific parameters to args.
    set_erm_training_parameters(args)

    model_name = get_model_name(args)
    log_dir = f"Teachers/{args.datamodule}/{model_name}/"
    cb_model = "CBERM/" if args.balance_erm else "biasedERM/"
    log_dir += cb_model

    # --- ERM TRAINING (NO HYPERPARAM SEARCH) ---
    curr_erm = get_current_erm(False)
    erm_version = curr_erm["version"]
    erm_metrics = curr_erm["metrics"]
    print(curr_erm)

    if erm_version == -1 or args.retrain:
        if args.retrain:
            print("====>>> Retraining")

        # use class-balanced sampler if balance_erm=True
        args.balanced_sampler = args.balance_erm
        model, erm_val_metrics, erm_test_metrics = main(
            args, model_class, args.datamodule_class
        )
        args.balanced_sampler = False

        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]

        curr_erm["version"] = erm_version
        curr_erm["metrics"] = erm_metrics
        print_metrics(erm_metrics, log_dir + "ERM")
        dump_erm(args, curr_erm)
        del model

    elif not erm_metrics:
        # Already have a version, but metrics missing – recompute them
        args.weights = get_weights(args, erm_version, idx=-1)
        args.eval_only = True
        _, erm_val_metrics, erm_test_metrics = main(
            args, model_class, args.datamodule_class
        )
        args.eval_only = False

        erm_metrics = [erm_val_metrics, erm_test_metrics]
        curr_erm["metrics"] = erm_metrics
        print_metrics(erm_metrics, log_dir + "ERM")
        dump_erm(args, curr_erm)
    else:
        # Model + metrics already exist
        print_metrics(erm_metrics, log_dir + "ERM")






    def print_results2(results, keys):
        return print_results(erm_metrics, results, keys)


    print("ERM", f"Version_{erm_version}", erm_metrics) 
    

    

    # Sets finetune types. Note that "group-unbalanced retraining" will be
    # either class-unbalanced or class-balanced based on the value
    # of args.balanced_sampler.

    def finetune_helper(
        finetune_type,
        dropout_prob=0,
        early_stop_num=None,
        finetune_lr=None,
        finetune_num_data=None,
        worst_group_pct=None,
    ):

        
        args.finetune_lr = finetune_lr if finetune_lr else args.lr

        param_str = ""
        if finetune_num_data:
            param_str += f" Num Data {finetune_num_data}"
        param_str += f" LR {args.finetune_lr}"
        if dropout_prob:
            param_str += f" Dropout {dropout_prob}"
        if early_stop_num:
            param_str += f" Early Stop {early_stop_num}"
        if worst_group_pct:
            param_str += f" Worst-group Pct {worst_group_pct}"
        print(f"{finetune_type.title()}{param_str}")

        early_stop_weights = None
        if early_stop_num:
            early_stop_weights = get_weights(args, erm_version, idx=early_stop_num-1)
        model, val_metrics, test_metrics = finetune_last_layer(
            args,
            finetune_type,
            model_class,
            dropout_prob=dropout_prob,
            early_stop_weights=early_stop_weights,
            finetune_num_data=finetune_num_data,
            worst_group_pct=worst_group_pct,
        )

        #print_metrics([val_metrics, test_metrics])  
        finetuned_version = model.trainer.logger.version
        finetuned_metrics = [val_metrics, test_metrics]

        # Save finetuned version
        dump_erm(args, {"version": finetuned_version, "metrics": finetuned_metrics}, debiased=True)

        """ display_type = finetune_type.replace(" ", "_").replace("-", "_")
        model_type = f"{args.model}"

        if args.model == "resnet":
            model_type +=f"_{args.resnet_version}"

        teacher_path = f"out/{args.datamodule}/{model_type}/{display_type}/teacher_{args.seed}.ckpt"
        model.trainer.save_checkpoint(teacher_path)  """ 

        return [val_metrics, test_metrics]
    
    #finetuned_erm = load_erm() 
    current_finetuned_erm = get_current_erm(True)  
    # Load Finetuned ERM model. 
    
    finetuned_model_version = current_finetuned_erm["version"]
    finetuned_metrics = current_finetuned_erm["metrics"]
    
    # Set DatataLoaders For last-layer Retraining
    set_training_parameters(args)

    # Finetune ERM model to mitigate biases
    #if finetuned_model_version == -1:  
    # Gets last-epoch ERM weights. From wich the finetune is applied
    args.weights = get_weights(args, erm_version, idx=-1)
    # Performs last-layer retraining.
    
    finetuned_metrics = finetune_helper(args.finetune_type)
    
    #elif not finetuned_metrics:  
    
    # Evaluate the the model already finetuned
    #args.weights = get_weights(args, finetuned_model_version, idx=-1)
    #args.eval_only = True
    #_, erm_val_metrics, erm_test_metrics = main(args, model_class, args.datamodule_class)
    #args.eval_only = False

    #finetuned_metrics = [erm_val_metrics, erm_test_metrics]
    #current_finetuned_erm["metrics"] = finetuned_metrics
    #dump_erm(args, current_finetuned_erm, debiased=True)
    
    print("FineTuned ERM", finetuned_metrics)
    display_type = args.finetune_type.replace(" ", "_").replace("-", "_") 
    print_metrics(finetuned_metrics, log_dir+display_type)
    #print_results2(results, finetune_types)

    
if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    finetune_types = [
        # "group-unbalanced retraining",
        "group-balanced retraining",
        """ "random self",
        "misclassification self",
        "early-stop misclassification self",
        "dropout disagreement self",
        "early-stop disagreement self", """
    ]

    finetune_types = ["group-balanced retraining"]

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    parser.add("--balance_erm", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to perform class-balancing during ERM training.")
    parser.add("--retrain", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to reset retraining if already trained.")
    parser.add("--balance_finetune", choices=["sampler", "subset", "none"], default="sampler",
               help="Which type of class-balancing to perform during finetuning.")
    parser.add("--finetune_lr", default=1e-2, type=float,
               help="Learning rate to finetune and mitigate bias.") 
    parser.add("--split", choices=["combined", "train"], default="train",
               help="The split to train on; either the train set or the combined train and held-out set.")
    parser.add("--train_pct", default=100, type=int,
               help="The percentage of the train set to utilize (for ablations)")
    
    parser.add("--finetune_num_data", default=500, type=int,
               help="Number of data used for finetune and debiasing")
    
    
    parser.add("--val_pct", default=100, type=int,
               help="The percentage of the val set to utilize (for ablations)") 
    parser.add("--finetune_type", choices=finetune_types, default="group-balanced retraining",
               help="The name of method used to debias the teacher models.")
    
#modification
    parser.add("--bootstrap", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to train using bootstrapped subsets of the dataset.")
    parser.add("--bootstrap_frac", default=0.8, type=float,
               help="Fraction of the dataset to sample for each bootstrap replicate.")
    parser.add("--bootstrap_n_sets", default=5, type=int,
               help="How many bootstrap replicates to train.")
    parser.add("--bootstrap_with_replacement", default=False, type=lambda x: bool(strtobool(x)),
               help="Sample with replacement (classic bootstrap) or without (unique 80%).")

#modification ends

    args = parser.parse_args()

    models, datamodules = valid_models_and_datamodules()

    #print(models[args.model])    
    experiment(args, models[args.model])

