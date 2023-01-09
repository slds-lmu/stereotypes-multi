
## Arguments

|short|default| help                                                                                                                                                                                                                             |
| :--- | :--- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`--batch-size`|`16`| Number of examples in a batch.                                                                                                                                                                                                   |
|`--device`|`cuda`| Device to run the training on (cpu or cuda).                                                                                                                                                                                     |
|`--datasets`|`None`| Specify the subset name of the huggingface Wikipedia dataset (e.g. 20220301.de, 20220301.en, 20220301.fr) to use in training. In case of 2 languages to use, provide them with a comma separation (e.g. 20220301.de,20220301.en) |
|`--pretrained-class`|`gpt2`| Choose the pretrained model to load from.                                                                                                                                                                                        |
|`--epochs`|`1`| Number of epochs in training.                                                                                                                                                                                                    |
|`--num-articles`|`None`| Number of articles to use per language. In case of a specific range will be given the beginning and end indexes should be given by separated with a comma (e.g. 100,150)                                                         |
|`--num-processes`|`None`| Number of processes to use in data creation (for multiprocessing).                                                                                                                                                               |
|`--max-seq-length`|`256`| Maximum token length allowed for the sentence pairs. Provides safety for exiting the program in the middle of training process due to memory exceeding reasons.                                                                  |
|`--core-lr`|`5e-06`| Learning rate for the core model.                                                                                                                                                                                                |
|`--head-lr`|`0.001`| Learning rate for the head model. This is not valid for T5 as it doesn't use any head to finetune itself.                                                                                                                        |
|`--tokenizer`|`GPT2Tokenizer`| Choose a tokenizer. Note that THUMT/mGPT model uses MT5Tokenizer.                                                                                                                                                                |
|`--load-dataset`|``| Provide the dataset name in saved_datasets folder to load the dataset that is saved by this program before. This helps to not create the same dataset in every training trial.                                                   |
|`--save-dataset`|``| Provide a name to save the dataset in the saved_datasets folder. This helps to not create the same dataset in every training trial.                                                                                              |
|`--load-model`|`None`| Provide the path of a model to load. This helps to continue training process or test a model.                                                                                                                                    |
|`--save-model`||Include it for the model to be saved.|
|`--wandb`|``| Provide the user name and project name by separating them with comma (e.g. user1,projA) to record the progress in Wandb.                                                                                                         |
|`--test`||Include it to do testing.|
|`--fp16`||To run the program with half precision.|
|`--accumulation-steps`|`1`| Number of batches to include in one gradient accumulation.                                                                                                                                                                       |
