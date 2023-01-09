from difflib import get_close_matches
import string
import pandas as pd


def find_correct_translation_intrasentence(row):
    correct_translation = get_close_matches(row["target"], row["context"].split(" "), cutoff=0.4)
    if len(correct_translation) > 0:
        correct_translation = correct_translation[0].strip(string.punctuation)
        return correct_translation
    else:
        return -1


def common_fix(df, test_type):
    df = df.replace("Dem Gefangenen", "Dem Häftling", regex=True)
    df = df.replace("Gefangener", "Häftling", regex=True)
    df = df.replace("Gefangenen", "Häftlinge", regex=True)
    df = df.replace("Gefangene", "Häftling", regex=True)

    df = df.replace("Schuljunge", "Schüler", regex=True)

    df = df.replace("Beweger", "Möbelpacker", regex=True)
    df = df.replace("Mover", "Möbelpacker", regex=True)
    df = df.replace("MOVER", "Möbelpacker", regex=True)
    df = df.replace("Umzugsunternehmen", "Möbelpacker", regex=True)

    df["target"].replace("Sierraleon", "Sierra Leone", regex=True, inplace=True)
    df["context"].replace("Sierra Leon", "Sierra Leone", regex=True, inplace=True)
    df["context"].replace("Sierraleon", "Sierra Leone", regex=True, inplace=True)
    df["context"].replace("sierra leon", "Sierra Leone", regex=True, inplace=True)

    df_problematic = df[df.apply(lambda x: (x.target.upper() not in x.context.upper()), axis=1)]
    # we split by whitespace in find_correct_translation_intrasentence() function,
    # this can be problematic for the targets which have 2 or more words,
    # so we should skip these targets
    for index, row in df_problematic.iterrows():
        if row["target"] in ["ihr selber", "sich selbst", "darstellende Künstler", "darstellenden Künstler",
                             "darstellender Künstler"]:
            continue
        correct_translation = find_correct_translation_intrasentence(row)
        if type(correct_translation) == str:
            df.loc[index]["target"] = correct_translation

    df_problematic2 = df[df.apply(lambda x: (x.target.upper() not in x.context.upper()), axis=1)]

    for index, row in df_problematic2.iterrows():
        if row["target"] == "männlich":
            df.loc[index]["target"] = "Mann"

    df_problematic3 = df[df.apply(lambda x: (x.target.upper() not in x.context.upper()), axis=1)]

    if test_type == "intrasentence":
        for index, row in df_problematic3.iterrows():
            if row["target"] == "herren":
                df.loc[index]["target"] = "Gentlement"

    else:
        for index, row in df_problematic3.iterrows():
            if row["target"] == "Kommandant":
                df.loc[index].replace("Befehlshaber", "Kommandant", regex=True, inplace=True)

    return df


def german_intrasentence_fix(df_intrasentence_de):
    df_intrasentence_de.replace("lieferant", "Zusteller", regex=True, inplace=True)
    df_intrasentence_de.replace("Lieferboten", "Zusteller", regex=True, inplace=True)
    df_intrasentence_de.replace("Lieferbote", "Zusteller", regex=True, inplace=True)

    df_intrasentence_de["target"].replace("darstellender Künstler", "darstellende Künstler", regex=True, inplace=True)
    df_intrasentence_de.loc[1233]["target"] = "darstellenden Künstler"
    df_intrasentence_de.loc[1933]["target"] = "darstellenden Künstler"

    # We dont know if it shd be "norwegische" (adj.) or "Norweger" (noun), it depends on the context
    # replace Norweigan by Norweg,
    # so it can match with the correct candidate versions (norwegische/Norweger) in the majority voting fix part
    df_intrasentence_de["target"].replace("Norweigan", "Norweg", regex=True, inplace=True)

    # get manual fixes
    fix1 = pd.read_excel('no_blank_de_urgent_fix.xlsx', index_col=0)
    fix2 = pd.read_excel('selber_problematic_intrasentence.xlsx', index_col=0)
    fix3 = pd.read_excel('de_all_data_fix.xlsx', index_col=0)

    fix1.index = fix1.index.astype(int)
    fix1.index.name = None
    fix3.index = fix3.index.astype(int)
    fix3.index.name = None
    fix3["is_fixed?"] = fix3["is_fixed?"].astype(int)
    fix3 = fix3[fix3["is_fixed?"] == 1]

    for index, row in fix1.iterrows():
        df_intrasentence_de.loc[index, "target"] = row["target"]
        if not pd.isna(row["context + BLANK"]):
            df_intrasentence_de.loc[index, "context"] = row["context + BLANK"]
        if not pd.isna(row["c1_correct"]):
            df_intrasentence_de.loc[index, "c1_word"] = row["c1_correct"]
        if not pd.isna(row["c2_correct"]):
            df_intrasentence_de.loc[index, "c2_word"] = row["c2_correct"]
        if not pd.isna(row["c3_correct"]):
            df_intrasentence_de.loc[index, "c3_word"] = row["c3_correct"]

    for index, row in fix2.iterrows():
        df_intrasentence_de.loc[index, "target"] = row["target"]
        # remove the problematic ones from the dataset
        if "problematic" in row["context"]:
            df_intrasentence_de = df_intrasentence_de[df_intrasentence_de.id != row["id"]]
        else:
            df_intrasentence_de.loc[index, "context"] = row["context"]

    for index, row in fix3[
        (fix3["context_correct"].isnull() == False) & (fix3["context_correct"] != "context already fixed")].iterrows():
        df_intrasentence_de.loc[index, "context"] = row["context_correct"]
    for index, row in fix3[fix3["c1_correct"].isnull() == False].iterrows():
        df_intrasentence_de.loc[index, "c1_word"] = row["c1_correct"]
    for index, row in fix3[fix3["c2_correct"].isnull() == False].iterrows():
        df_intrasentence_de.loc[index, "c2_word"] = row["c2_correct"]
    for index, row in fix3[fix3["c3_correct"].isnull() == False].iterrows():
        df_intrasentence_de.loc[index, "c3_word"] = row["c3_correct"]

    return common_fix(df_intrasentence_de, "intrasentence")


def german_intersentence_fix(df_intersentence_de):
    df_intersentence_de["target"].replace("darstellender Künstler", "darstellende Künstler", regex=True, inplace=True)
    df_intersentence_de.loc[816]["target"] = "darstellenden Künstlers"
    df_intersentence_de.loc[920]["target"] = "darstellenden Künstler"
    df_intersentence_de.loc[1393]["target"] = "darstellenden Künstler"
    df_intersentence_de.loc[1553]["target"] = "darstellender Künstler"

    intersentence_problematic = df_intersentence_de[
        df_intersentence_de.apply(lambda x: x.target.upper() not in x.context.upper(), axis=1)]
    for index, row in intersentence_problematic.iterrows():
        if row["target"] == "lieferant":
            df_intersentence_de.loc[index].replace("lieferanten", "Zusteller", regex=True, inplace=True)
            df_intersentence_de.loc[index].replace("lieferant", "Zusteller", regex=True, inplace=True)
            df_intersentence_de.loc[index].replace("Lieferboten", "Zusteller", regex=True, inplace=True)
            df_intersentence_de.loc[index].replace("Lieferbote", "Zusteller", regex=True, inplace=True)

    # get manual fixes
    fix1 = pd.read_excel('selber_problematic_intersentence.xlsx', index_col=0)
    for index, row in fix1.iterrows():
        df_intersentence_de.loc[index, "target"] = row["target"]
        df_intersentence_de.loc[index, "context"] = row["context"]

    return common_fix(df_intersentence_de, "intersentence")
