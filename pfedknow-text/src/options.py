import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    #For pFedMe
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    # federated arguments

    parser.add_argument('--epochs', type = int, default = 100,
                        help = "number of rounds of training")
    parser.add_argument('--num_users', type = int, default = 100,
                        help = "number of users: K")
    parser.add_argument('--frac', type = float, default = 0.1,
                        help = 'the fraction of clients:C')
    parser.add_argument('--local_ep', type = int, default = 3,
                        help = 'the number of local epochs: E')
    parser.add_argument('--local_bs', type = int, default = 8,
                        help = 'local batch size: B')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        help = 'learning rate')
    parser.add_argument('--momentum', type = float, default = 0.5,
                        help = 'SGD momentum (default = 0.5)')
    


    # model arguments
    
    parser.add_argument('--model_G', type = str, default = 'BERT', help = 'model name')
    parser.add_argument('--model', type=str, default='TINYBERT', help='model name')
    parser.add_argument('--kernel_num', type = int, default = 9, help = 'number of each kind of kernel')
    parser.add_argument('--kernel_size', type = str, default = '3,4,5', help = 'comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type = int, default = 1, help = 'number of channels in imgs')
    parser.add_argument('--norm', type = str, default = 'batch_norm', help = 'batch_norm, layer_norm, none')
    parser.add_argument('--num_filters', type = int, default = 32, help = 'number of filters')
    parser.add_argument('--max_pool', type = str, default = 'True', help = 'whether use max pooling rather than strided con')

    # other arguments

    parser.add_argument('--dataset', type = str, default = 'yahoo', help = 'name of dataset')
    parser.add_argument('--num_classes', type = int, default = 10, help = 'number of classes')
    parser.add_argument('--gpu', default = 1, help = 'to use cuda, set to a specific gpu id, default set to use cpu')
    parser.add_argument('--gpuid', default = "cuda:1", help = 'gpu id')
    # parser.add_argument('--optimizer', type = str, default = 'sgd', help = 'type of optimizer')
    parser.add_argument('--iid', type = int, default = 1, help = 'default set to iid, 0 for non-iid')
    parser.add_argument('--unequal', type = int, default = 0, help = "whether to use unequal data splits for non-iid data, 0 for equal split")
    parser.add_argument('--stopping_rounds', type = int, default = 10, help = 'rounds of early stopping')
    parser.add_argument('--verbose', type = int, default = 1, help = 'verbose')
    parser.add_argument('--seed', type = int, default = 1, help = 'random seed')
    parser.add_argument('--label_rate', type=float, default=0.1, help="the fraction of labeled data")
    parser.add_argument("--log_dir", help="dir for log file;", type=str, default="logs")
    parser.add_argument("--log_fn", help="file name;", type=str, default="output")
    parser.add_argument('--distill_round', type=int, default=1, help="Round of multiteacher distillation")
    parser.add_argument('--heat-epochs', type=int, default=1, help='Number of preheat epoches')

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other "
             "data files) for the task."
    )
    parser.add_argument(
        "--bert_model", default=None, type=str, required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-large-cased, "
             "bert-base-multilingual-uncased, bert-base-multilingual-cased, "
             "bert-base-chinese."
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model "
             "predictions and checkpoints will be written."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after "
             "WordPiece tokenization. \nSequences longer than this"
             " will be truncated, and sequences shorter \n"
             "than this will be padded."
    )
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="Run all steps with a small model and sample data."
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_prune",
        action='store_true',
        help="Whether to run pruning."
    )
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_anal",
        action='store_true',
        help="Whether to run analyzis on the diagnosis set (for NLI model)."
    )
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available"
    )
    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     default=42,
    #     help="random seed for initialization"
    # )
    # parser.add_argument(
    #     "--verbose",
    #     action='store_true',
    #     help="Print data examples"
    # )
    parser.add_argument(
        "--no-progress-bars",
        action='store_true',
        help="Disable progress bars"
    )
    parser.add_argument(
        "--feature_mode",
        action='store_true',
        help="Don't update the BERT weights."
    )
    parser.add_argument(
        "--toy_classifier",
        action='store_true',
        help="Toy classifier"
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Don't raise an error when the output dir exists"
    )
    parser.add_argument(
        "--toy_classifier_n_heads",
        default=1,
        type=int,
        help="Number of heads in the simple (non-BERT) sequence classifier"
    )
    #arguments for prunning
    return parser



def training_args(parser):
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training."
    )
    # train_group.add_argument(
    #     "--learning_rate",
    #     default=5e-5,
    #     type=float,
    #     help="The initial learning rate for Adam."
    # )
    train_group.add_argument(
        "--attn_dropout",
        default=0.1,
        type=float,
        help="Head dropout rate"
    )
    train_group.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    train_group.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear "
        "learning rate warmup for. "
        "E.g., 0.1 = 10%% of training."
    )
    train_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    train_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before "
        "performing a backward/update pass."
    )


def pruning_args(parser):
    prune_group = parser.add_argument_group("Pruning")
    prune_group.add_argument(
        "--compute_head_importance_on_subset",
        default=1.0,
        type=float,
        help="Percentage of the training data to use for estimating "
        "head importance."
    )
    prune_group.add_argument(
        "--prune_percent",
        default=[50],
        type=float,
        nargs="*",
        help="Percentage of heads to prune."
    )
    prune_group.add_argument(
        "--prune_number",
        default=None,
        nargs="*",
        type=int,
        help="Number of heads to prune. Overrides `--prune_percent`"
    )
    prune_group.add_argument(
        "--prune_reverse_order",
        action='store_true',
        help="Prune in reverse order of importance",
    )
    prune_group.add_argument(
        "--normalize_pruning_by_layer",
        action='store_true',
        help="Normalize importance score by layers for pruning"
    )
    prune_group.add_argument(
        "--actually_prune",
        action='store_true',
        help="Really prune (like, for real)"
    )
    prune_group.add_argument(
        "--at_least_x_heads_per_layer",
        type=int,
        default=0,
        help="Keep at least x attention heads per layer"
    )
    prune_group.add_argument(
        "--exact_pruning",
        action='store_true',
        help="Reevaluate head importance score before each pruning step."
    )
    prune_group.add_argument(
        "--eval_pruned",
        action='store_true',
        help="Evaluate the network after pruning"
    )
    prune_group.add_argument(
        "--n_retrain_steps_after_pruning",
        type=int,
        default=0,
        help="Retrain the network after pruning for a fixed number of steps"
    )
    prune_group.add_argument(
        "--retrain_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for retraining the network after pruning for a "
        "fixed number of steps"
    )
    prune_group.add_argument(
        "--retrain_pruned_heads",
        action='store_true',
        help="Retrain the pruned heads"
    )
    prune_group.add_argument(
        "--n_retrain_steps_pruned_heads",
        type=int,
        default=0,
        help="Retrain the pruned heads"
    )
    prune_group.add_argument(
        "--reinit_from_pretrained",
        action='store_true',
        help="Reinitialize the pruned head from the pretrained model"
    )
    prune_group.add_argument(
        "--no_dropout_in_retraining",
        action='store_true',
        help="Disable dropout when retraining heads"
    )
    prune_group.add_argument(
        "--only_retrain_val_out",
        action='store_true',
        help="Only retrain the value and output layers for attention heads"
    )


def eval_args(parser):
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Total batch size for eval."
    )
    eval_group.add_argument(
        "--attention_mask_heads", default="", type=str, nargs="*",
        help="[layer]:[head1],[head2]..."
    )
    eval_group.add_argument(
        '--reverse_head_mask',
        action='store_true',
        help="Mask all heads except those specified by "
        "`--attention-mask-heads`"
    )
    eval_group.add_argument(
        '--save-attention-probs', default="", type=str,
        help="Save attention to file"
    )


def analysis_args(parser):
    anal_group = parser.add_argument_group("Analyzis")
    anal_group.add_argument(
        "--anal_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the "
        "`diagnistic-full.tsv` file."
    )


def fp16_args(parser):
    fp16_group = parser.add_argument_group("FP16")
    fp16_group.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of"
        " 32-bit"
    )
    fp16_group.add_argument(
        '--loss_scale',
        type=float, default=0,
        help="Loss scaling to improve fp16 numeric stability. "
        "Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n"
    )

    