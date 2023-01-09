import os
import torch
from datasets import concatenate_datasets, load_from_disk
from torch import nn, autocast
from torch.cuda.amp import GradScaler
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from tqdm import tqdm
import data_loader
from torch.utils.data import DataLoader
import sys
import transformers
import numpy as np
from sklearn.metrics import accuracy_score
import time
import wandb
#import paramiko

sys.path.append("..")
from models import models
from inference import count_parameters

np.random.seed(27)
torch.manual_seed(27)
torch.cuda.manual_seed(27)
torch.cuda.manual_seed_all(27)
transformers.logging.set_verbosity_error()


def parse_args():
    pretrained_model_choices = ['gpt2', 'gpt2-medium', 'gpt2-large', 'miguelvictor/multilingual-gpt2-large',
                                'THUMT/mGPT', "t5-base", "t5-small", 'google/mt5-base', 'google/mt5-small',
                                "GermanT5/t5-efficient-gc4-german-base-nl36", "flax-community/spanish-t5-small",
                                'plguillou/t5-base-fr-sum-cnndm']
    args = ArgumentParser()
    args.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run the training on (cpu or cuda).")

    args.add_argument("--data-name", default=None, type=str, help="Specify the subset name of the huggingface Wikipedia dataset (e.g. 20220301.de, 20220301.en, 20220301.fr) to use in training. In case of 2 languages to use, provide them with a comma separation (e.g. 20220301.de,20220301.en)")
    args.add_argument("--data-cache-dir", default=None, type=str, help='Specify the cached directory address of subset.')
    args.add_argument("--num-articles", help="Number of articles to use per language. In case of a specific range will be given the beginning and end indexes should be given by separated with a comma (e.g. 100,150)")
    args.add_argument("--num-processes", default=None, type=int, help="Number of processes to use in data creation (for multiprocessing).")
    args.add_argument("--load-dataset", default="", help="Provide the dataset name in saved_datasets folder to load the dataset that is saved by this program before. This helps to not create the same dataset in every training trial.")
    args.add_argument("--save-dataset", default="", help="Provide a name to save the dataset in the saved_datasets folder. This helps to not create the same dataset in every training trial.")

    args.add_argument("--pretrained-class", default="gpt2", choices=pretrained_model_choices, help="Choose the pretrained model to load from.")
    args.add_argument("--batch-size", default=16, type=int, help="Number of examples in a batch.")
    args.add_argument("--epochs", default=1, type=int, help="Number of epochs in training.")
    args.add_argument("--max-seq-length", default=256, type=int, help="Maximum token length allowed for the sentence pairs. Provides safety for exiting the program in the middle of training process due to memory exceeding reasons.")
    args.add_argument("--core-lr", default=5e-6, type=float, help="Learning rate for the core model.")
    args.add_argument("--head-lr", default=1e-3, type=float, help="Learning rate for the head model. This is not valid for T5 as it doesn't use any head to finetune itself.")
    args.add_argument("--tokenizer", default="GPT2Tokenizer", help="Choose a tokenizer. Note that THUMT/mGPT model uses MT5Tokenizer.")
    args.add_argument("--test", default=False, action="store_true", help="Include it to do testing.")
    args.add_argument("--fp16", default=False, action="store_true", help="To run the program with half precision.")
    args.add_argument("--accumulation-steps", type=int, default=1, help="Number of batches to include in one gradient accumulation.")

    args.add_argument("--load-model", default=None, help="Provide the path of a model to load. This helps to continue training process or test a model.")
    args.add_argument("--save-model", default=False, action="store_true", help="Include it for the model to be saved.")
    args.add_argument("--save-model-dir", default='../models/finetuned_models/', type=str, help="Specify the directory address for saving models.")
    args.add_argument("--delete-prev-models", default=False, action="store_true", help="Delete previously saved models to save space")
    args.add_argument("--wandb", default="", help="Provide the user name and project name by separating them with comma (e.g. user1,projA) to record the progress in Wandb.")
    return args.parse_args()


def main(args):
    torch.manual_seed(27)

    device = args.device
    fp16 = args.fp16
    pretrained_class = args.pretrained_class
    batch_size = args.batch_size
    core_lr = args.core_lr
    head_lr = args.head_lr
    num_articles = args.num_articles if args.num_articles else args.load_dataset.split("_")[-1]

    # Get Tokenizer
    tokenizer = getattr(transformers, args.tokenizer).from_pretrained(pretrained_class)

    # Get Model
    model = getattr(models, "ModelNSP")(pretrained_class, tokenizer).to(device)
    model.core_model.output_past = False
    if args.test:
        model.eval()
    else:
        model.train()
    print(f"Number of parameters: {count_parameters(model)}")

    # this enables us to do batched training, GPT2 wasn't trained with a padding token.
    if "gpt2" in args.tokenizer.lower():
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.core_model.resize_token_embeddings(len(tokenizer))

    # Get Dataset
    load_dataset = args.load_dataset
    if load_dataset != "":
        dataset = load_from_disk(os.path.join("saved_datasets", load_dataset))
    else:
        datasets = [dataset for dataset in args.data_name.split(',')]
        dataset = data_loader.create_dataset(datasets[0], num_articles, args.data_cache_dir,
                                             num_proc=args.num_processes)
        print("Dataset creation is done for", datasets[0])
        print("Number of examples in", datasets[0], ":", len(dataset))
        if len(datasets) > 1:
            for dataset_name in datasets[1:]:
                dataset_sub = data_loader.create_dataset(dataset_name, num_articles, args.data_cache_dir,
                                                         num_proc=args.num_processes)
                print("Dataset creation is done for", dataset_name)
                print("Number of examples in", dataset_name, ":", len(dataset_sub))
                dataset = concatenate_datasets([dataset, dataset_sub]).shuffle(seed=27)
        if args.save_dataset != "":
            if not os.path.exists("saved_datasets"):
                os.makedirs("saved_datasets")
            dataset.save_to_disk(os.path.join("saved_datasets", args.save_dataset))
    print("Total number of examples:", len(dataset))

    def tokenize_examples(example):
        sentence_pairs = []
        labels = []
        language = example[0]["language"]
        for item in example:
            if 't5' in pretrained_class.lower() and 'mt5' not in pretrained_class.lower():
                if language == "english":
                    sentence_pairs.append(("binary classification: " + item["text"], item["next_sentence"]))
                elif language == "german":
                    sentence_pairs.append(("binäre Klassifikation: " + item["text"], item["next_sentence"]))
                elif language == "spanish":
                    sentence_pairs.append(("clasificación binaria: " + item["text"], item["next_sentence"]))
                elif language == "french":
                    sentence_pairs.append(("classification binaire: " + item["text"], item["next_sentence"]))
            else:
                sentence_pairs.append((item["text"], item["next_sentence"]))
            labels.append(item["label"])
        encoded_dict = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentence_pairs, add_special_tokens=True,
                                                   truncation="longest_first", padding=True,
                                                   return_tensors="pt", return_token_type_ids=True,
                                                   return_attention_mask=True, max_length=args.max_seq_length)
        encoded_dict["labels"] = torch.LongTensor(labels)
        return encoded_dict
    # to see in dataframe format
    # dataset.set_format("pandas")
    # dataset_df = dataset[:]

    dataset = dataset.remove_columns(['id', 'url', 'title'])

    num_workers = 0 if args.num_processes is None else args.num_processes
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False,
                            collate_fn=tokenize_examples, num_workers=num_workers)

    # the pretrained model has been fairly optimized, while the NSP head has been randomly initialized.
    # using different learning rates helps speed up training.
    specific_learning_rates = [{"params": model.core_model.parameters(), "lr": core_lr, "correct_bias": False},
                               {"params": model.nsp_head.parameters(), "lr": head_lr, "correct_bias": False}]
    if 't5' in pretrained_class.lower():
        specific_learning_rates = [{"params": model.core_model.parameters(), "lr": core_lr, "correct_bias": False}]

    optimizer = transformers.AdamW(specific_learning_rates, lr=core_lr, correct_bias=False)

    print(f"Device is set to {device}!")

    model = nn.DataParallel(model)

    test_scores = []
    accumulation_steps = args.accumulation_steps
    num_training_steps = (len(dataloader) // accumulation_steps) * args.epochs
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=num_training_steps//100+1, num_training_steps=num_training_steps)
    # Also try
    # scheduler = ReduceLROnPlateau(optimizer, "max", patience=10, verbose=True)

    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Total Training Steps: {num_training_steps}")

    if not args.test:
        save_model_dir = args.save_model_dir
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        save_path = os.path.join(save_model_dir, pretrained_class.replace("/", "_") + "_" + num_articles)
    if args.wandb != "":
        wandb.init(project=args.wandb.split(",")[1], entity=args.wandb.split(",")[0])
        wandb.config = {
            "epochs": args.epochs,
            "batch_size": batch_size,
            "accumulation steps": accumulation_steps,
            "number of sentences": len(dataset)
        }

    scaler = None
    # Main training loop
    if fp16:
        scaler = GradScaler()

    if args.test:
        checkpoint = torch.load(args.load_model, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

    elif args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_prev = checkpoint['epoch']
        train_batch_num_prev = checkpoint['train_batch_num']
        scaler = checkpoint["scaler"]

    start_time = time.time()
    for epoch in range(args.epochs):
        if not args.test and args.load_model is not None and epoch < epoch_prev:
            continue
        running_loss = 0.0
        running_accuracy = 0.0
        ticks = 0.0
        number_of_batches = len(dataloader)
        for train_batch_num, example in tqdm(enumerate(dataloader), total=len(dataloader)):
            # skip the previous batches
            if not args.test and args.load_model is not None and train_batch_num <= train_batch_num_prev and epoch == epoch_prev:
                continue
            # the next batch
            elif not args.test and args.load_model is not None and train_batch_num == train_batch_num_prev+1 and epoch == epoch_prev:
                running_loss = checkpoint['running_loss']
                running_accuracy = checkpoint['running_accuracy']
                ticks = checkpoint['ticks']

            input_ids = example["input_ids"]
            token_type_ids = example["token_type_ids"]
            attention_mask = example["attention_mask"]
            labels = example["labels"]

            if device == "cuda":
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            if 't5' in pretrained_class.lower() and not args.test:
                # convert integer labels to string due to T5's seq2seq methodology
                encoded_labels_list = []
                for label in labels:
                    encoded_label = model.module.find_label_encoding(str(label.item()), tokenizer)
                    encoded_labels_list.append(encoded_label)
                label_input_ids = torch.cat((tuple(encoded_labels_list)), dim=0)

                if fp16:
                    with autocast(device_type=device):
                        loss = model(input_ids, attention_mask=attention_mask, labels=label_input_ids)
                        output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                       labels=labels, test=2)
                else:
                    loss = model(input_ids, attention_mask=attention_mask, labels=label_input_ids)
                    output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                   labels=labels, test=2)
                output_probs = output.softmax(dim=-1)
                predictions = torch.argmax(output_probs, dim=-1)

            else:
                if fp16:
                    with autocast(device_type=device):
                        output, loss = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                             labels=labels, test=args.test, device=device)
                else:
                    output, loss = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                         labels=labels, test=args.test, device=device)
                # output is (batch_size x 2)
                output_probs = output.softmax(dim=-1)
                predictions = torch.argmax(output_probs, dim=-1)

            loss = loss.mean(dim=0)
            loss = loss / accumulation_steps
            running_loss += loss.item()

            accuracy = accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy())
            if args.test:
                test_scores.append(accuracy)
            running_accuracy += accuracy
            ticks += 1.0

            if not args.test:
                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Gradient accumulation: Instead of updating the network weights on every batch,
                # we can save gradient values, proceed to the next batch and add up the new gradients.
                # The weight update is then done only after several batches have been processed by the model.
                # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
                if train_batch_num % accumulation_steps == 0:
                    if fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if fp16:
                        scaler.step(optimizer)
                        scheduler.step()
                        scaler.update()
                    else:
                        optimizer.step()
                        scheduler.step()
                    model.zero_grad()

            # print and monitor the results every 500 examples
            # It is important to monitor it on the last batch (final result) too
            if ((train_batch_num+1) * args.batch_size) % 500 == 0 or (train_batch_num+1) == number_of_batches:
                for param_group in optimizer.param_groups:
                    print("LR:", param_group['lr'])
                acc = (running_accuracy / ticks)
                loss = (running_loss / ticks) * accumulation_steps
                progress = (train_batch_num+1) / number_of_batches
                if args.wandb != "" and 't5' not in pretrained_class.lower():
                    wandb.log({"loss": loss, "accuracy": acc, "progress": round(progress*100,2),
                               "core_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"]})
                elif args.wandb != "":
                    wandb.log({"loss": loss, "accuracy": acc, "progress": round(progress * 100, 2),
                               "core_lr": optimizer.param_groups[0]["lr"]})
                print(f"[Epoch {epoch+1}: {progress*100:.2f}%] Accuracy: {acc}, Loss: {loss}")
                print("Time passed until the batch", train_batch_num, time.time() - start_time)
                # save the model every 10 percent of the dataset, so you would have 10 checkpoints
                running_loss = 0.0
                running_accuracy = 0.0
                ticks = 0.0

            # save the model every 10 percent of the dataset
            # +1 is added to smooth the division to avoid getting zero in denominator (laplace smoothing)
            if not args.test and args.save_model and (train_batch_num+1) % (number_of_batches // 10 + 1) == 0:
                if args.delete_prev_models == True:
                    test = os.listdir(save_model_dir)
                    for item in test:
                        if item.endswith(".pth"):
                            os.remove(os.path.join(save_model_dir, item))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_batch_num': train_batch_num,
                    'running_loss': running_loss,
                    'running_accuracy': running_accuracy,
                    'ticks': ticks,
                    "scaler": scaler
                }, save_path+f"_{train_batch_num}_{epoch}.pth")

    if args.test:
        print(f"Final test accuracy: {np.mean(test_scores)}")

    if not args.test and args.save_model:
        if args.delete_prev_models == True:
            test = os.listdir(save_model_dir)
            for item in test:
                if item.endswith(".pth"):
                    os.remove(os.path.join(save_model_dir, item))
        save_path = save_path + f"_{core_lr}_{head_lr}_{args.epochs}_FINAL.pth"
        print(f"Saving model to {save_path}")
        torch.save({'model_state_dict': model.state_dict()}, save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
