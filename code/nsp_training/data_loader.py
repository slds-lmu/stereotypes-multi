from torch.utils.data import Dataset
import random
random.seed(27)
import nltk
nltk.download('punkt')
import os
import psutil
process = psutil.Process(os.getpid())

from datasets import load_dataset, Dataset, concatenate_datasets


def create_related_examples(examples_related, num_proc):
    examples_related = examples_related.filter(lambda example: example['id'] == example['next_id'], num_proc=num_proc)
    examples_related = examples_related.remove_columns(['next_id'])
    return examples_related


def create_unrelated_examples(examples_unrelated, num_proc):
    last_index = len(examples_unrelated)-1

    def find_unrelated_sentence(example):
        unrelated_example = examples_unrelated[random.randint(0, last_index)]
        while unrelated_example["id"] == example["id"]:
            unrelated_example = examples_unrelated[random.randint(0, last_index)]
        example["next_sentence"] = unrelated_example["text"]
        example["label"] = 0
        return example
    examples_unrelated = examples_unrelated.map(find_unrelated_sentence, num_proc=num_proc)
    return examples_unrelated


def explode_dataset(dataset):
    dataset.set_format("pandas")
    dataset = dataset[:]
    dataset = dataset.explode('text').reset_index(drop=True)
    examples_unrelated = Dataset.from_pandas(dataset)
    dataset["next_sentence"] = dataset[["text"]][1:].append({"text": ""}, ignore_index=True)
    dataset["next_id"] = dataset[["id"]][1:].append({"id": ""}, ignore_index=True)
    dataset = Dataset.from_pandas(dataset)
    print("\nMemory Consumption explode_dataset:", process.memory_full_info().uss * 1e-9)
    return examples_unrelated, dataset


def tokenize_articles(article):
    article["text"] = nltk.sent_tokenize(article["text"], language=article["language"])
    return article


def load_hf_dataset(dataset_name, num_articles, cache_dir):
    data_date = dataset_name.split('.')[0]
    data_language = dataset_name.split('.')[1]
    try:
        dataset = load_dataset("wikipedia", dataset_name, cache_dir=cache_dir)["train"]
    except:
        dataset = load_dataset("wikipedia", language=data_language, date=data_date, cache_dir=cache_dir)["train"]
    if "," in num_articles:
        num_articles = num_articles.split(",")
        dataset = dataset.shuffle(seed=27).select(range(int(num_articles[0]), int(num_articles[1])))
    else:
        dataset = dataset.shuffle(seed=27).select(range(int(num_articles)))
    if data_language == "en":
        language = "english"
    elif data_language == "de":
        language = "german"
    elif data_language == "es":
        language = "spanish"
    elif data_language == "fr":
        language = "french"
    elif data_language == "tr":
        language = "turkish"
    length = len(dataset)
    dataset = dataset.add_column("language", [language] * length)
    dataset = dataset.add_column("next_sentence", [""] * length)
    dataset = dataset.add_column("label", [1] * length)
    return dataset


def create_dataset(dataset_name, num_articles, cache_dir, num_proc=None):
    random.seed(27)
    print("\nLoading the dataset..")
    dataset = load_hf_dataset(dataset_name, num_articles, cache_dir)
    print("\nTokenizing articles to list of sentences..")
    dataset = dataset.map(tokenize_articles, num_proc=num_proc)
    # here it converts HF dataset to pandas df to make explode()
    print("\nExploding dataset using pandas df")
    examples_unrelated, examples_related = explode_dataset(dataset)
    print("\nMemory Consumption after explode:", process.memory_full_info().uss*1e-9)
    print("\nCreating unrelated (random next sentence) examples..")
    examples_unrelated = create_unrelated_examples(examples_unrelated, num_proc=num_proc)
    print("\nCreating related (real next sentence) examples..")
    examples_related = create_related_examples(examples_related, num_proc)
    examples = concatenate_datasets([examples_related, examples_unrelated]).shuffle(seed=27)
    return examples


if __name__ == "__main__":
    nsp = create_dataset("20220301.en", "220000", 8)
