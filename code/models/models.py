import transformers
import torch
from torch import nn


class BertLM(transformers.BertPreTrainedModel):
    def __init__(self):
        pass

    def __new__(self, pretrained_model, tokenizer):
        return transformers.BertForMaskedLM.from_pretrained(pretrained_model)

class BertNextSentence(transformers.BertPreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model, tokenizer):
        return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model)

class GPT2LM(transformers.GPT2PreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model, tokenizer):
        return transformers.GPT2LMHeadModel.from_pretrained(pretrained_model)

class T5LM(transformers.T5PreTrainedModel):
    def __init__(self):
        pass

    def __new__(self, pretrained_model, tokenizer):
        return transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model)

class mT5LM(transformers.T5PreTrainedModel):
    def __init__(self):
        pass

    def __new__(self, pretrained_model, tokenizer):
        return transformers.MT5ForConditionalGeneration.from_pretrained(pretrained_model)

class ModelNSP(nn.Module):
    def __init__(self, pretrained_model, tokenizer, nsp_dim=300):
        super(ModelNSP, self).__init__()
        if "gpt" in pretrained_model.lower():
            base_model_name = "gpt2"
        elif "mt5" in pretrained_model.lower():
            base_model_name = "mt5"
            self.zero_token = self.find_label_encoding("0", tokenizer).item()
            self.one_token = self.find_label_encoding("1", tokenizer).item()
        elif "t5" in pretrained_model.lower():
            base_model_name = "t5"
            self.zero_token = self.find_label_encoding("0", tokenizer).item()
            self.one_token = self.find_label_encoding("1", tokenizer).item()
        else:
            base_model_name = None
        self.pretrained2model = {"gpt2": "GPT2Model", "t5": "T5ForConditionalGeneration",
                                 "mt5": "MT5ForConditionalGeneration"}
        self.model_class = self.pretrained2model[base_model_name]
        self.core_model = getattr(transformers, self.model_class).from_pretrained(pretrained_model)
        self.core_model.train()
        hidden_size = self.core_model.config.hidden_size
        # remove nsp head for t5
        self.nsp_head = nn.Sequential(nn.Linear(hidden_size, nsp_dim), nn.Linear(nsp_dim, nsp_dim), nn.Linear(
            nsp_dim, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, test=0, device="cpu"):
        if 'GPT2' in self.model_class:
            outputs = self.core_model(input_ids, attention_mask=attention_mask)
            # sum up all values for each vocab dimension
            output = outputs[0].mean(dim=1)
            logits = self.nsp_head(output)
        elif 'T5' in self.model_class:
            if test != 0:
                outputs = self.core_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=3,
                                                   output_scores=True, return_dict_in_generate=True)
                logits = self.find_logits_t5(outputs, device, self.zero_token, self.one_token)
            else:
                outputs = self.core_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return outputs.loss
        else:
            outputs = self.core_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = self.nsp_head(outputs[1])

        if labels is not None and test != 2:
            loss = self.criterion(logits, labels)
            if type(logits) == tuple:
                logits = logits[0]
            return logits, loss
        return logits

    @staticmethod
    def find_label_encoding(input_str, tokenizer):
        encoded_str = tokenizer.encode(input_str, add_special_tokens=False, return_tensors="pt")
        if encoded_str.size(dim=1) == 2:  # sometimes the T5 tokenizers add the token 3 strangely
            encoded_str = torch.index_select(encoded_str, 1, torch.tensor([1]))
        return encoded_str

    @staticmethod
    def find_logits_t5(outputs, device, zero_token, one_token):
        scores = outputs.scores[1]
        # for the first example in the batch
        logits = torch.Tensor([scores[0][zero_token], scores[0][one_token]]).unsqueeze(0)
        # for the next examples in the batch
        for i in range(1, len(scores)):
            logits = torch.cat(
                (logits, torch.Tensor([scores[i][zero_token], scores[i][one_token]]).unsqueeze(0)), dim=0)
        if device == "cuda":
            logits = logits.cuda()
        return logits
