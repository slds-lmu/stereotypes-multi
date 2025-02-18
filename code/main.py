from argparse import ArgumentParser
import os
from inference import main as inference_main
from evaluation import main as evaluation_main

def parse_args():
    """ Parses the command line arguments. """
    pretrained_model_choices = ['bert-base-uncased', 'bert-base-cased', "bert-large-uncased-whole-word-masking",
                                'bert-large-uncased', 'bert-large-cased', "bert-base-multilingual-cased",
                                'gpt2', 'gpt2-medium', 'gpt2-large',
                                't5-small', 't5-base', 't5-large', 'THUMT/mGPT', 'google/mt5-base',
                                'dbmdz/bert-base-german-cased', 'bert-base-german-cased', 'dbmdz/bert-base-turkish-cased',
                                'dbmdz/german-gpt2', 'redrussianarmy/gpt2-turkish-cased',
                                'GermanT5/t5-efficient-gc4-german-base-nl36',
                                'flax-community/spanish-t5-small', 'dccuchile/bert-base-spanish-wwm-cased','PlanTL-GOB-ES/gpt2-base-bne',
                                'plguillou/t5-base-fr-sum-cnndm', 'flaubert/flaubert_base_cased', 'asi/gpt-fr-cased-small']
    tokenizer_choices = ["BertTokenizer", "GPT2Tokenizer", "T5Tokenizer", "MT5Tokenizer", "AutoTokenizer", "GPT2TokenizerFast"]
    parser = ArgumentParser()
    parser.add_argument("--skip-inference", help="Skip inference and use pre-calculated predictions.",
                        default=False, action="store_true")
    parser.add_argument("--predictions-file")
    parser.add_argument("--output-file")
    parser.add_argument("--pretrained-class", default="bert-base-cased", choices=pretrained_model_choices,
                        help="Choose the pretrained model to load from.")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--intrasentence-df-file", type=str, help="Choose the intrasentence dataframe to predict on.")
    parser.add_argument("--intersentence-df-file", type=str, help="Choose the intersentence dataframe to predict on.")
    parser.add_argument("--skip-intrasentence", help="Skip intrasentence predictions.",
                        default=False, action="store_true")
    parser.add_argument("--intrasentence-model", type=str, default='BertLM', choices=[
                        'BertLM', 'GPT2LM', "T5LM", "mT5LM"],
                        help="Choose a model architecture for the intrasentence task.")
    parser.add_argument("--skip-intersentence",
                        default=False, action="store_true", help="Skip intersentence predictions.")
    parser.add_argument("--intersentence-model", type=str, default='BertNextSentence', choices=[
        'BertNextSentence', 'ModelNSP', 'GPT2LM', 'T5LM', 'mT5LM'], help="Choose the model for the intersentence task.")
    parser.add_argument("--nsp-model-path", type=str, default=None, help="Provide the path for the trained NSP model.")
    parser.add_argument("--tokenizer-name", type=str, default='BertTokenizer', choices=tokenizer_choices,
                        help="Choose a string tokenizer.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of examples in a batch.")
    parser.add_argument("--num-workers", type=int, default=1, help="Choose number of processes to run togetherly (multiprocessing).")
    parser.add_argument("--case-no", type=str, default="d", help="Choose the case no for intersentence generative calculations for GPT-2 based models.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    language = args.intrasentence_df_file[-6:-4] if args.intrasentence_df_file is not None \
        else args.intersentence_df_file[-6:-4]
    print(args.predictions_file)
    if args.skip_inference or os.path.exists(args.predictions_file):
        print("Predictions already exist, re-using them!")
        evaluation_main(args)
    else:
        print("Predictions dont exist, inferencing them!")
        inference_main(args)
        # do the evaluation here using the predictions file that created by the line above