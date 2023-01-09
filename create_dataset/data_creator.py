import boto3
import pandas as pd
import os
import string
from argparse import ArgumentParser

from german_translation_fixes import german_intrasentence_fix, german_intersentence_fix


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", help="Path for the English StereoSet Dataset", default="data/dev.json")
    parser.add_argument("--aws-access-key-id")
    parser.add_argument("--aws-secret-access-key")
    parser.add_argument("--language", default="en", help="Language of the target dataset (e.g. de, es)")
    return parser.parse_args()


def normalize_df(df):
    sentences = pd.json_normalize(df["sentences"])
    df["c1_sentence"] = sentences.apply(lambda row: row[0]["sentence"], axis=1)
    df["c1_gold_label"] = sentences.apply(lambda row: row[0]["gold_label"], axis=1)
    df["c2_sentence"] = sentences.apply(lambda row: row[1]["sentence"], axis=1)
    df["c2_gold_label"] = sentences.apply(lambda row: row[1]["gold_label"], axis=1)
    df["c3_sentence"] = sentences.apply(lambda row: row[2]["sentence"], axis=1)
    df["c3_gold_label"] = sentences.apply(lambda row: row[2]["gold_label"], axis=1)
    df["c1_id"] = sentences.apply(lambda row: row[0]["id"], axis=1)
    df["c1_labels"] = sentences.apply(lambda row: row[0]["labels"], axis=1)
    df["c2_id"] = sentences.apply(lambda row: row[1]["id"], axis=1)
    df["c2_labels"] = sentences.apply(lambda row: row[1]["labels"], axis=1)
    df["c3_id"] = sentences.apply(lambda row: row[2]["id"], axis=1)
    df["c3_labels"] = sentences.apply(lambda row: row[2]["labels"], axis=1)
    df = df.drop("sentences", 1)
    return df


def add_candidate_word(df_intrasentence):
    for index, example in df_intrasentence.iterrows():
        word_idx = None
        # find where the BLANK word is
        for idx, word in enumerate(example['context'].split(" ")):
            if "BLANK" in word:
                word_idx = idx
        # find the candidate word and add it to the original dataset as a new column
        for i in range(1, 4):
            i = str(i)
            candidate_word = example["c" + i + "_sentence"].split(" ")[word_idx]
            candidate_word = candidate_word.translate(str.maketrans('', '', string.punctuation))
            df_intrasentence.loc[index, "c" + i + "_word"] = candidate_word
    return df_intrasentence


def translate_df(df, columns_to_translate, client, language='de'):
    # TODO: Check if you can improve this loop
    for i, row in df.iterrows():
        for col in columns_to_translate:
            df.iloc[i][col] = client.translate_text(
                Text=row[col], SourceLanguageCode='en', TargetLanguageCode=language)["TranslatedText"]
    return df


def intrasentence_context_translate(df_intrasentence_de, client, language):
    terminology = ('en,'+language+'\nBLANK,BLANK').encode('ascii')
    client.import_terminology(
        Name='blank',
        MergeStrategy='OVERWRITE',
        Description='blankForContext',
        TerminologyData={
            'File': terminology,
            'Format': 'CSV'
        }
    )
    for i, row in df_intrasentence_de.iterrows():
        df_intrasentence_de.iloc[i]["context"] = client.translate_text(
            Text=row["context"], SourceLanguageCode='en', TargetLanguageCode=language,
            TerminologyNames=["blank"])["TranslatedText"]

    df_intrasentence_de.replace("LEERES", "BLANK", regex=True, inplace=True)
    df_intrasentence_de.replace("LEER", "BLANK", regex=True, inplace=True)
    return df_intrasentence_de


def german_translation_fixes(df_intrasentence_de, df_intersentence_de):
    def change_football_spieler(df):
        df = df.replace("Football-Spieler", "American-Football-Spieler", regex=True)
        df = df.replace("Fußballspieler", "American-Football-Spieler", regex=True)
        df = df.replace("Footballspieler", "American-Football-Spieler", regex=True)
        return df

    df_intrasentence_de["c1_word"] = df_intrasentence_de["c1_word"].replace("[^A-Za-z0-9äÄöÖüÜß ]", "", regex=True)
    df_intrasentence_de["c2_word"] = df_intrasentence_de["c2_word"].replace("[^A-Za-z0-9äÄöÖüÜß ]", "", regex=True)
    df_intrasentence_de["c3_word"] = df_intrasentence_de["c3_word"].replace("[^A-Za-z0-9äÄöÖüÜß ]", "", regex=True)

    # 'Football-Spieler' and 'Fußballspieler' to 'Footballspieler'
    df_intrasentence_de = change_football_spieler(df_intrasentence_de)
    df_intersentence_de = change_football_spieler(df_intersentence_de)

    df_intrasentence_de["context"] = df_intrasentence_de["context"].replace("-BLANK", " BLANK", regex=True)
    df_intrasentence_de["context"] = df_intrasentence_de["context"].replace("BLANK-", "BLANK ", regex=True)

    df_intrasentence_de = german_intrasentence_fix(df_intrasentence_de)
    df_intersentence_de = german_intersentence_fix(df_intersentence_de)

    return df_intrasentence_de, df_intersentence_de


def data_creator(data_path, aws_access_key_id, aws_secret_access_key, language="en"):
    df = pd.read_json(data_path)["data"]
    df_intrasentence = normalize_df(pd.json_normalize(df["intrasentence"]))
    df_intersentence = normalize_df(pd.json_normalize(df["intersentence"]))
    df_intrasentence = add_candidate_word(df_intrasentence)

    if language == "en":
        df_intrasentence.to_pickle("data/intrasentence/df_intrasentence_en.pkl")
        df_intersentence.to_pickle("data/intersentence/df_intersentence_en.pkl")
        exit()
    client = boto3.client('translate',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name='eu-west-1')
    intrasentence_translate_columns = ["target", "c1_word", "c2_word", "c3_word"]
    intersentence_translate_columns = ["context", "target", "c1_sentence", "c2_sentence", "c3_sentence"]
    df_intrasentence_new = df_intrasentence.copy()
    df_intersentence_new = df_intersentence.copy()

    df_intrasentence_new = translate_df(df_intrasentence_new, intrasentence_translate_columns, client, language)
    df_intersentence_new = translate_df(df_intersentence_new, intersentence_translate_columns, client, language)

    df_intrasentence_new = intrasentence_context_translate(df_intrasentence_new, client, language)

    df_intrasentence_new.to_pickle("data/intrasentence/df_intrasentence_temp_" + args.language + ".pkl")
    df_intersentence_new.to_pickle("data/intersentence/df_intersentence_temp_" + args.language + ".pkl")

    # uncomment below if you already have the translated dataframe
    #df_intrasentence_new = pd.read_pickle("data/intrasentence/df_intrasentence_de_v2_0.pkl")
    #df_intersentence_new = pd.read_pickle("data/intersentence/df_intersentence_de_v1_0.pkl")

    # keep storing the original targets
    df_intrasentence_new["target_original"] = df_intrasentence["target"]
    df_intersentence_new["target_original"] = df_intersentence["target"]

    if language == "de":
        df_intrasentence_new, df_intersentence_new = german_translation_fixes(
            df_intrasentence_new, df_intersentence_new)

    # create the candidate sentences for intrasentence
    for index, example in df_intrasentence_new.iterrows():
        # replace BLANK with the candidate word
        for i in range(1, 4):
            c_sent = example["context"].replace("BLANK", example["c" + str(i) + "_word"])
            df_intrasentence_new.loc[index, "c" + str(i) + "_sentence"] = c_sent

    return df_intrasentence_new, df_intersentence_new


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir('data/intrasentence'):
        os.mkdir("data/intrasentence")
    if not os.path.isdir('data/intersentence'):
        os.mkdir("data/intersentence")
    df_intrasentence_main, df_intersentence_main = data_creator(
        args.data, args.aws_access_key_id, args.aws_secret_access_key, args.language)
    df_intrasentence_main.to_pickle("data/intrasentence/df_intrasentence_" + args.language + ".pkl")
    df_intersentence_main.to_pickle("data/intersentence/df_intersentence_" + args.language + ".pkl")
