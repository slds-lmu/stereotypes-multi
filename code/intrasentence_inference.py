from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import time

import data_loader


def discriminative_inference(df_intrasentences, model, device, batch_size, tokenizer, num_workers):
    # t5 masked token: https: // bytemeta.vip / repo / agemagician / ProtTrans / issues / 39
    mask_token = "<extra_id_0>" if "t5" in tokenizer.name_or_path.lower() else tokenizer.mask_token
    mask_token_idx = tokenizer.encode(mask_token, add_special_tokens=False)
    assert len(mask_token_idx) == 1
    mask_token_idx = mask_token_idx[0]
    loader = data_loader.load_data(df_intrasentences, tokenizer, "discriminative_intrasentence", batch_size,
                                   num_workers, mask_token)
    word_probabilities = defaultdict(list)
    print("Probability calculations starts for intrasentence")
    start_time = time.time()
    # calculate the logits for each prediction
    # for sentence_id, next_token, input_ids, attention_mask in tqdm(loader, total=len(loader)):
    for batch_num, tokens_dict in tqdm(enumerate(loader), total=len(loader)):
        # start by converting everything to a tensor
        input_ids, attention_mask, masked_tokens, candidate_ids = tokens_dict
        if device == "cuda":
            input_ids, attention_mask = data_loader.move_cuda([input_ids, attention_mask])
        if "t5" in model.name_or_path.lower():
            """
             the generator output for T5 is always "beginning_token, <extra_id_0>, masked_word"
             the length of this masked word depends on max_length, we keep it as 3 to get only one word for masked word
             in this case the scores output would have 2 elements, the first one is probability distribution for
             <extra_id_0>, and the second one is the probability distribution for the masked word, which is
             what we are looking for.
            """
            output = model.generate(input_ids, attention_mask=attention_mask, max_length=3, output_scores=True,
                                    return_dict_in_generate=True).scores[1].softmax(dim=-1)
            for idx, item in enumerate(output):
                word_probabilities[candidate_ids[idx]].append(item[masked_tokens[idx]].item())
        else:
            mask_idxs = (input_ids == mask_token_idx)
            # get the probability vectors (size of vocab_size) for every word in the candidate sentence
            output = model(input_ids, attention_mask=attention_mask)[0].softmax(dim=-1)
            # extract the probability vector of only the masked word
            output = output[mask_idxs]
            masked_tokens = torch.tensor(masked_tokens).cuda() if device == "cuda" else torch.tensor(masked_tokens)
            output = output.index_select(1, masked_tokens).diag()
            for idx, item in enumerate(output):
                word_probabilities[candidate_ids[idx]].append(item.item())
    print("Probability calculations finished in: ", time.time()-start_time)

    sentence_probabilities = []
    for k, v in word_probabilities.items():
        # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
        score = np.mean(v)
        pred = {'id': k, 'score': score}
        sentence_probabilities.append(pred)
    return sentence_probabilities


def generative_inference(df_intrasentences, model, device, tokenizer, batch_size, num_workers):
    start_token = "<|endoftext|>"
    start_token = tokenizer.encode(start_token, return_tensors="pt").to(device)
    initial_token_probabilities = model(start_token)[0].softmax(dim=-1)[0][0]
    loader = data_loader.load_data(df_intrasentences, tokenizer, "generative_intrasentence", batch_size, num_workers)
    predictions = []
    print("Probability calculations starts for intrasentence")
    start_time = time.time()
    for batch_num, tokens_dict in tqdm(enumerate(loader), total=len(loader)):
        input_ids, attention_mask, candidate_ids = tokens_dict
        if device == "cuda":
            input_ids, attention_mask = data_loader.move_cuda([input_ids, attention_mask])

        output = model(input_ids, attention_mask=attention_mask)[0].softmax(dim=-1)
        # now output includes probabilities for each word in the candidate sentences,
        # for each word it has the probability to be followed with a word X.
        # There are 50257 Xs (vocab size in tokenizer)
        for sent_no, item in enumerate(output):
            # finds the probability of first word
            joint_sentence_probability = [initial_token_probabilities[input_ids[sent_no][0]].item()]
            # find the probability of each word to follow the previous word
            for idx in range(1, attention_mask[sent_no].sum().item()):
                joint_sentence_probability.append(item[idx - 1, input_ids[sent_no][idx]].item())
            # sum all the words' probabilities to find the probability of the "sentence"
            score = np.sum([np.log2(j) for j in joint_sentence_probability]) / len(joint_sentence_probability)
            score = np.power(2, score)
            probabilities = {'id': candidate_ids[sent_no], 'score': score}
            predictions.append(probabilities)

    print("Probability calculations finished in: ", time.time() - start_time)
    return predictions
