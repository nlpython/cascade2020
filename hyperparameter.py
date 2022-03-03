import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default='clue2020ner', type=str,
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--train_file_path", default='./clue/train.json', type=str,
                        help="The path of trainning file.", )
    parser.add_argument("--dev_file_path", default='./clue/dev.json', type=str,
                        help="The path of trainning file.")
    parser.add_argument("--bert_path", default='./bert-base-chinese', type=str,
                        help="Path to pre-trained models or shortcut name selected in the list: " )
    parser.add_argument("--log_output_dir", default='./log', type=str,
                        help="The output directory where the logs which created in trainning.", )
    parser.add_argument("--checkpoint_dir", default='./checkpoint', type=str,
                        help="The output directory where the the models predictions and checkpoints will be written.")


    # Other parameters
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])

    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Set this flag if you are using an uncased models.")

    # adversarial training
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--bert_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for bert layers.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--other_learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for other layers.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )


    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument("--id2ordertag", default=['X', 'B', 'I', 'O', '[CLS]', '[SEP]'], type=list,
                        help="The mapping relation from id to order tag.")
    parser.add_argument("--num_orders", default=6, type=int, help="The number of order tags.")
    parser.add_argument("--id2entitytag", default=['X', 'O', 'address', 'book', 'company', 'game', 'government', 'movie',
                                                   'name', 'organization', 'position', 'scene', '[CLS]', '[SEP]'], type=list,
                        help="The mapping relation from id to entity tag.")
    parser.add_argument("--num_entitys", default=14, type=int, help="The number of entity tags.")
    parser.add_argument("--id2tag", default=['X', 'O', '[CLS]', '[SEP]', 'I', 'B-address', 'B-book', 'B-company', 'B-game', 'B-government',
                                             'B-movie', 'B-name', 'B-organization', 'B-position', 'B-scene'], type=list)



    return parser.parse_args()

args = get_argparse()

from torch import nn
encoder = nn.TransformerEncoderLayer(768, nhead=8, batch_first=True)
transformer = nn.TransformerEncoder(num_layers=6, encoder_layer=encoder)

