# Inference

After following steps below, the new predictions json file will be created in the `results/predictions` folders.

## Arguments For Inference

| arg                       |default|help|
|:--------------------------| :--- | :--- |
| `--pretrained-class`      |`bert-base-cased`|Choose the pretrained model to load from.|
| `--device`                |`cpu`|`None`|
| `--intrasentence-df-file` |`None`|Choose the intrasentence dataframe to predict on.|
| `--intersentence-df-file` |`None`|Choose the intersentence dataframe to predict on.|
| `--skip-intrasentence`    ||Skip intrasentence predictions.|
| `--intrasentence-model`   |`BertLM`|Choose a model architecture for the intrasentence task.|
| `--skip-intersentence`    ||Skip intersentence predictions.|
| `--intersentence-model`   |`BertNextSentence`|Choose the model for the intersentence task.|
| `--nsp-model-path`        |`None`|Provide the path for the trained NSP model.|
| `--tokenizer-name`        |`BertTokenizer`|Choose a string tokenizer.|
| `--batch-size`            |`1`|Number of examples in a batch.|
| `--num-workers`           |`1`|Choose number of processes to run togetherly (multiprocessing).|
| `--case-no`               |`d`|Choose the case no for intersentence generative calculations for GPT-2 based models.|

###case-no Argument

For GPT-2 intersentence generative calculations, there is not only one way of calculating the predictions and
the methodology depends on the user's preference.

Consider the intersentences as A and B, where sentence A is the context sentence and sentence B is the candidate sentence that follows the sentence A.
The program calculates following language modelling scores for each candidate sentence:
Score of context sentence: score(A)
Score of candidate sentence **without** inputting it to the model with sentence A: score(B)
Score of candidate sentence **with** inputting it to the model with sentence A: score(B|A)
Score of full sentence: score(A ∩ B)

The cases are following:

"orig":
score(B)/score(A).
This is the version that [StereoSet](https://github.com/moinnadeem/stereoset) uses, as a consequence of that, it is called "orig" to abbreaviate the word "original".
This case doesn't consider the scores of the sentences togetherly, therefore, it doesn't really computes their dependence (consecutiveness).
Thus, this case **doesn't provide satisfying results at all**, it is only added as an option for comparability purposes.

"c":
score(A ∩ B).
This version considers the sentences togetherly as B follows A and computes the score according to.
It's mathematically and logically sensible approach and it usually gives highest ICAT scores.

**"d":**
score(B|A).
**This case is more preferred than the others** because T5 models calculates the scores very similar to this approach, hence, it allows to compare 2 models in a fair environment.
This version computes the scores only from the probabilities of sentence B, however,
the probabilities of sentence B is queried to the model by combining it with sentence A.
Therefore, its extremely similar approach with case no "c" and gives similar results as it is also extremely logical and mathematically working approach.

"e":
score(B|A)/score(B).
This version outputs very unstable results, for some models it shows extremely large scores, and for some it shows extremely low.
Therefore, it is **not recommended** to conclude the results by using only this case.

"f":
score(A ∩ B)/score(B).
Similar approach with case e and gives usually similar results, **not recommended**.

# Evaluation

After following steps below, the final evaluations will be printed and can be saved in json or excel formats.

* Provide the predictions file (--predictions argument) that includes the probability predictions for each candidate
* Provide intrasentence or intersentence files (or both) (--intrasentence-df-file and --intersentence-df-file) that
includes the dataset in pandas format which is the output created by `create_dataset/` folder.
* Include --skip-intrasentence or --skip-intersentence to skip a test
* To save the outputs to a file, provide a path for the file to the --output-file argument,
the program will append the results if the file already exists, otherwise it will create a new file.
Please provide it with ".json" or an excel extension (e.g. ".xlsx").
