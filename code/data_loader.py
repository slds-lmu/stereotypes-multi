from torch.utils import data


class DataLoader(object):
    def __init__(self, df, tokenizer, inference_type, mask_token, pretrained_class):
        self.tokenizer = tokenizer
        self.sentences = []
        self.mask_token = mask_token
        self.pretrained_class = pretrained_class

        for index, row in df.iterrows():
            for i in range(1, 4):
                if "t5" in inference_type or "gpt" in inference_type:
                    inference_type = '_'.join(inference_type.split("_")[:2])
                eval("self.add_" + inference_type + "_candidates(row, str(i))")

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

    def add_discriminative_intrasentence_candidates(self, example, c_no):
        insertion_tokens = self.tokenizer.encode(example["c" + c_no + "_word"], add_special_tokens=False)
        for idx in range(len(insertion_tokens)):
            insertion = self.tokenizer.decode(insertion_tokens[:idx])
            new_sentence = example["context"].replace("BLANK", f"{insertion}{self.mask_token}")
            self.sentences.append({"sentence": new_sentence, "candidate_id": example["c" + c_no + "_id"],
                                   "masked_token": insertion_tokens[idx]})

    def add_discriminative_intersentence_candidates(self, example, c_no):
        self.sentences.append({"sentence_pair": (example["context"], example["c"+c_no+"_sentence"]),
                               "candidate_id": example["c"+c_no+"_id"]})

    def add_generative_intrasentence_candidates(self, example, c_no):
        self.sentences.append({"candidate": example["c" + c_no + "_sentence"],
                               "candidate_id": example["c" + c_no + "_id"]})

    def add_generative_intersentence_candidates(self, example, c_no):
        context = example["context"]
        if context[-1] not in [".", "!", "?"]:
            context = f"{context}."
        self.sentences.append({"context": context, "candidate": example["c" + c_no + "_sentence"],
                               "candidate_id": example["c" + c_no + "_id"]})

    def tokenize_discriminative_intrasentence(self, sent_dicts):
        sentences, candidate_ids, masked_tokens = [], [], []
        for sent_dict in sent_dicts:
            sentences.append(sent_dict["sentence"])
            candidate_ids.append(sent_dict["candidate_id"])
            masked_tokens.append(sent_dict["masked_token"])
        tokens_dict = self.tokenizer.batch_encode_plus(sentences, truncation="longest_first", padding=True,
                                                       return_attention_mask=True, return_tensors="pt")
        return tokens_dict["input_ids"], tokens_dict["attention_mask"], masked_tokens, candidate_ids

    def tokenize_discriminative_intersentence(self, sent_dicts):
        sentence_pairs, candidate_ids = [], []
        for sent_dict in sent_dicts:
            if 't5' in self.pretrained_class.lower() and 'mt5' not in self.pretrained_class.lower():
                lst = list(sent_dict["sentence_pair"])
                sent_dict["sentence_pair"] = tuple(["binary classification: " + lst[0], lst[1]])
            sentence_pairs.append(sent_dict["sentence_pair"])
            candidate_ids.append(sent_dict["candidate_id"])
        tokens_dict = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentence_pairs,
                                                       truncation="longest_first", padding=True, return_tensors="pt",
                                                       return_token_type_ids=True, return_attention_mask=True)
        return tokens_dict["input_ids"], tokens_dict["attention_mask"], tokens_dict["token_type_ids"], candidate_ids

    def tokenize_generative_intrasentence(self, sent_dicts):
        sentences, candidate_ids = [], []
        for sent_dict in sent_dicts:
            sentences.append(sent_dict["candidate"])
            candidate_ids.append(sent_dict["candidate_id"])
        tokens_dict = self.tokenizer.batch_encode_plus(sentences, truncation="longest_first", padding=True,
                                                       return_attention_mask=True, return_tensors="pt")
        return tokens_dict["input_ids"], tokens_dict["attention_mask"], candidate_ids

    def tokenize_generative_intersentence_gpt(self, sent_dicts):
        full_sentences, context_lengths, candidates, candidate_ids = [], [], [], []
        for sent_dict in sent_dicts:
            full_sentences.append(sent_dict["context"] + " " + sent_dict["candidate"])
            # also store the length of the context to understand where context ends
            context_lengths.append(len(self.tokenizer.encode(sent_dict["context"])))
            candidates.append(sent_dict["candidate"])
            candidate_ids.append(sent_dict["candidate_id"])
        full_sent_tokens_dict = self.tokenizer.batch_encode_plus(
            full_sentences, truncation="longest_first", padding=True, return_attention_mask=True, return_tensors="pt")
        candidate_tokens_dict = self.tokenizer.batch_encode_plus(
            candidates, truncation="longest_first", padding=True, return_attention_mask=True, return_tensors="pt")

        return full_sent_tokens_dict["input_ids"], full_sent_tokens_dict["attention_mask"],\
            candidate_tokens_dict["input_ids"], candidate_tokens_dict["attention_mask"], candidate_ids, context_lengths

    def tokenize_generative_intersentence_t5(self, sent_dicts):
        encoder_input, decoder_input, candidate_ids = [], [], []
        for sent_dict in sent_dicts:
            encoder_input.append(sent_dict["context"] + " <extra_id_0>")
            # The model will automatically create the decoder_input_ids based on the labels, by shifting them one
            # position to the right and prepending the config.decoder_start_token_id,
            # which for T5 is equal to 0 (i.e. the id of the pad token).
            # We need to use at decoder_input_ids instead of labels due to having attention mask
            decoder_input.append("<pad> <extra_id_0> " + sent_dict["candidate"])
            candidate_ids.append(sent_dict["candidate_id"])
        encoder_tokens_dict = self.tokenizer.batch_encode_plus(
            encoder_input, truncation="longest_first", padding=True, return_attention_mask=True, return_tensors="pt")
        decoder_tokens_dict = self.tokenizer.batch_encode_plus(
            decoder_input, truncation="longest_first", padding=True, return_attention_mask=True, return_tensors="pt")

        return encoder_tokens_dict["input_ids"], encoder_tokens_dict["attention_mask"],\
            decoder_tokens_dict["input_ids"], decoder_tokens_dict["attention_mask"], candidate_ids


def load_data(df_intersentences, tokenizer, func_name, batch_size, num_workers, mask_token=None, pretrained_class=None):
    dataset = DataLoader(df_intersentences, tokenizer, func_name, mask_token, pretrained_class)
    loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=eval("dataset.tokenize_" + func_name),
                             num_workers=num_workers)
    return loader


def move_cuda(variables):
    for i in range(len(variables)):
        variables[i] = variables[i].cuda()
    return tuple(variables)
