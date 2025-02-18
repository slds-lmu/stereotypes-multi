import json
import os
from argparse import ArgumentParser
import pandas as pd
import torch
import transformers

from models import models
import intrasentence_inference
import intersentence_inference


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_df_init(model_name, pretrained_class, device, tokenizer_name, df_file, nsp_model_path=None):
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(pretrained_class, padding_side="right")
    print("tokenizer created")
    model = getattr(models, model_name)(pretrained_class, tokenizer).to(device)
    print(f"Number of parameters: {count_parameters(model):,}")

    if "gpt2" in type(tokenizer).__name__.lower():
        print("Adding <PAD> token to tokenizer...")
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.core_model.resize_token_embeddings(len(tokenizer)) if nsp_model_path is not None else \
            model.base_model.resize_token_embeddings(len(tokenizer))

    if nsp_model_path is not None:
        model = torch.nn.DataParallel(model)
        try:
            model.load_state_dict(torch.load(nsp_model_path, map_location=torch.device(device))['model_state_dict'])
        except (Exception,):
            model.load_state_dict(torch.load(nsp_model_path, map_location=torch.device(device)))
    model.eval()
    df = pd.read_pickle(df_file)
    return model, tokenizer, df


def inference(args):
    results = {}
    device, batch_size, num_workers, case_no = args.device, args.batch_size, args.num_workers, args.case_no
    if not args.skip_intrasentence:
        model_intrasentence, tokenizer, df_intrasentence = model_df_init(
            args.intrasentence_model, args.pretrained_class, device, args.tokenizer_name, args.intrasentence_df_file)
        if args.intrasentence_model == "GPT2LM":
            results["intrasentence"] = intrasentence_inference.generative_inference(
                df_intrasentence, model_intrasentence, device, tokenizer, batch_size, num_workers)
        else:
            results["intrasentence"] = intrasentence_inference.discriminative_inference(
                df_intrasentence, model_intrasentence, device, batch_size, tokenizer, num_workers)

    if not args.skip_intersentence:
        model_intersentence, tokenizer, df_intersentence = model_df_init(
            args.intersentence_model, args.pretrained_class, device, args.tokenizer_name, args.intersentence_df_file,
            args.nsp_model_path)
        if "gpt" in args.intersentence_model.lower():
            # intersentence GPT based without NSP
            results["intersentence"] = intersentence_inference.generative_inference_gpt(
                df_intersentence, model_intersentence, case_no, device, tokenizer, batch_size, num_workers)
        elif "t5" in args.intersentence_model.lower():
            # intersentence T5 based without NSP
            results["intersentence"] = intersentence_inference.generative_inference_t5(
                df_intersentence, model_intersentence, device, tokenizer, batch_size, num_workers)
        else:
            # intersentence discriminative approaches: GPT2 or T5 based with NSP or BERT-based models
            results["intersentence"] = intersentence_inference.discriminative_inference(
                df_intersentence, model_intersentence, device, batch_size, tokenizer, num_workers,
                args.pretrained_class)
    return results

def main(args):
    inference(args)
    language = args.intrasentence_df_file[-6:-4] if args.intrasentence_df_file is not None \
        else args.intersentence_df_file[-6:-4]
    if not os.path.isdir(os.path.join("..", "results", 'predictions')):
        os.mkdir(os.path.join("..", "results", 'predictions'))
    intersentence_model_name = args.intersentence_model + "_" + args.case_no \
        if (args.intersentence_model == "GPT2LM" or args.intersentence_model == "T5LM") else args.intersentence_model
    intersentence_model_name = "" if args.skip_intersentence is True else intersentence_model_name
    intrasentence_model_name = "" if args.skip_intrasentence is True else args.intrasentence_model
    pretrained_class_name = args.pretrained_class.replace("/", "_")
    result_path = os.path.join("..", "results", "predictions",
                               pretrained_class_name + "_" + intrasentence_model_name + "_" +
                               intersentence_model_name + "_" + language + ".json")
    with open(result_path, "w+") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    results = main(args)
