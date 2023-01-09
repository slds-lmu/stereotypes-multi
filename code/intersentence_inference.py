import torch
from tqdm import tqdm
import numpy as np
import time

import data_loader


def discriminative_inference(df_intersentence, model, device, batch_size, tokenizer, num_workers, pretrained_class):
    loader = data_loader.load_data(df_intersentence, tokenizer, "discriminative_intersentence", batch_size,
                                   num_workers, pretrained_class=pretrained_class)
    print("Probability calculations started for intersentence")
    start_time = time.time()
    predictions = []
    for batch_num, batch in tqdm(enumerate(loader), total=len(loader)):
        input_ids, attention_mask, token_type_ids, sentence_ids = batch
        if device == "cuda":
            input_ids, attention_mask, token_type_ids = data_loader.move_cuda([input_ids, attention_mask,
                                                                               token_type_ids])
        if "t5" in pretrained_class.lower() or "mt5" in pretrained_class.lower():
            outputs = model.module.core_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=3,
                                                       output_scores=True, return_dict_in_generate=True)
            outputs = model.module.find_logits_t5(outputs, device, model.module.zero_token, model.module.one_token)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = outputs.logits if "bert" in pretrained_class.lower() else outputs
        outputs = torch.softmax(outputs, dim=-1)

        for idx in range(input_ids.shape[0]):
            probabilities = {'id': sentence_ids[idx],
                             'score': outputs[idx, 0].item() if "bert" in pretrained_class.lower() else outputs[
                                 idx, 1].item()}
            predictions.append(probabilities)
    print("Probability calculations finished for intersentence in: ", time.time() - start_time)
    return predictions


def generative_inference_gpt(df_intersentences, model, case_no, device, tokenizer, batch_size, num_workers):
    unconditional_start_token = "<|endoftext|>"
    start_token = tokenizer.encode(unconditional_start_token, return_tensors='pt').to(device)
    initial_token_probabilities = model(start_token)[0].softmax(dim=-1)[0][0]
    loader = data_loader.load_data(df_intersentences, tokenizer, "generative_intersentence_gpt", batch_size,
                                   num_workers)
    predictions = []
    for batch_num, tokens_dict in tqdm(enumerate(loader), total=len(loader)):
        full_sent_input_ids, full_sent_attention_mask, candidate_input_ids, candidate_attention_mask, \
            candidate_ids, context_len = tokens_dict
        if device == "cuda":
            full_sent_input_ids, full_sent_attention_mask, candidate_input_ids, candidate_attention_mask = data_loader.\
                move_cuda([full_sent_input_ids, full_sent_attention_mask, candidate_input_ids,
                           candidate_attention_mask])
        # we use the 0th item since that corresponds to the prediction scores over vocab tokens
        # all tokens (context + candidate)
        output = model(full_sent_input_ids, attention_mask=full_sent_attention_mask)[0].softmax(dim=-1)
        if case_no in ["orig", "e", "f"]:
            only_candidate_output = model(candidate_input_ids, attention_mask=candidate_attention_mask)[0].softmax(
                dim=-1)
        for sent_no, item in enumerate(output):
            first_token_prob = initial_token_probabilities[full_sent_input_ids[sent_no][0]].item()
            context_probability = [first_token_prob]
            candidate_probability = []
            full_sentence_probability = [first_token_prob]
            # iterate over the sentence and setup those probabilities.
            for idx in range(0, full_sent_attention_mask[sent_no].sum().item() - 1):
                # 0th output corresponds to the probability of the 1st token.
                prob = item[idx, full_sent_input_ids[sent_no][idx + 1]].item()
                if idx < context_len[sent_no] - 1:
                    context_probability.append(prob)
                else:
                    candidate_probability.append(prob)
                full_sentence_probability.append(prob)

            if case_no in ["orig", "e", "f"]:
                only_candidate_probability = [initial_token_probabilities[candidate_input_ids[sent_no][0]].item()]
                # setup the probability for the sentence if we didn't provide the context
                for idx in range(0, candidate_attention_mask[sent_no].sum().item() - 1):
                    only_candidate_probability.append(
                        only_candidate_output[sent_no][idx, candidate_input_ids[sent_no][idx + 1]].item())

            if case_no == "orig":  # (selected)
                context_score = np.power(2, np.mean([np.log2(i) for i in context_probability]))
                only_candidate_score = np.power(2, np.mean([np.log2(i) for i in only_candidate_probability]))
                overall_score = only_candidate_score / context_score
            elif case_no == "b":  # candidate / whole sentence
                sentence_score = np.power(2, np.mean([np.log2(i) for i in candidate_probability]))
                full_sentence_score = np.power(2, np.mean([np.log2(i) for i in full_sentence_probability]))
                overall_score = sentence_score / full_sentence_score
            elif case_no == "c":  # (main_selected)
                full_sentence_score = np.power(2, np.mean([np.log2(i) for i in full_sentence_probability]))
                overall_score = full_sentence_score
            elif case_no == "d": # (selected)
                sentence_score = np.power(2, np.mean([np.log2(i) for i in candidate_probability]))
                overall_score = sentence_score
            elif case_no == "e":  # (selected)
                sentence_score = np.power(2, np.mean([np.log2(i) for i in candidate_probability]))
                only_candidate_score = np.power(2, np.mean([np.log2(i) for i in only_candidate_probability]))
                overall_score = sentence_score / only_candidate_score
            else:  # case_no == "f"
                full_sentence_score = np.power(2, np.mean([np.log2(i) for i in full_sentence_probability]))
                only_candidate_score = np.power(2, np.mean([np.log2(i) for i in only_candidate_probability]))
                overall_score = full_sentence_score / only_candidate_score

            probabilities = {'id': candidate_ids[sent_no], 'score': overall_score}
            predictions.append(probabilities)
    return predictions


def generative_inference_t5(df_intersentences, model, device, tokenizer, batch_size, num_workers):
    loader = data_loader.load_data(df_intersentences, tokenizer, "generative_intersentence_t5", batch_size, num_workers)
    predictions = []
    for batch_num, tokens_dict in tqdm(enumerate(loader), total=len(loader)):
        encoder_input_ids, encoder_attention_mask, decoder_input_ids,\
            decoder_attention_mask, candidate_ids = tokens_dict
        if device == "cuda":
            encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask = data_loader.\
                move_cuda([encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask])

        output = model(encoder_input_ids, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input_ids,
                       decoder_attention_mask=decoder_attention_mask).logits.softmax(dim=-1)
        for sent_no, item in enumerate(output):
            sent_probs = []
            # iterate over the sentence and setup those probabilities.
            # the last token is EOS </s> and we don't need to take the probability of it
            for idx in range(1, decoder_attention_mask[sent_no].sum().item() - 1):
                # 0th output corresponds to the probability of the 1st token.
                prob = item[idx, decoder_input_ids[sent_no][idx+1]].item()
                sent_probs.append(prob)
            sentence_score = np.power(2, np.mean([np.log2(i) for i in sent_probs]))
            probabilities = {'id': candidate_ids[sent_no], 'score': sentence_score}
            predictions.append(probabilities)
    return predictions
