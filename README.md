# How Different is Stereotypical Bias in Different Languages? Analysis of Multilingual Language Models
> I. Tolga Ozturk, Rostislav Nedelchev, Dr. Matthias Assenmacher, Prof. Christian Heumann

This repository contains an extensible codebase to measure stereotypical bias on pretrained models, both in monolingual and multilingual models with different languages (e.g. English, German), as well as code to replicate our results. 
It allows you to create your own dataset in the language that you are interested in, and then obtain the bias evaluation for that language on your preferred pre-trained model. 
The repository assists you to conduct an analysis in various pretrained models in different architectures, encoder based (e.g. BERT), decoder based (e.g. GPT-2), encoder-decoder based (e.g. T5).

## Install
```bash
git clone https://github.com/slds-lmu/stereotypes-multi.git
cd stereotypes-multi
python -m pip install -r requirements.txt
```

## Experiments
There are 3 main steps to be done:

### 1. Creating your own dataset
One can create their own dataset in the language that they interested in.
We have provided English and German versions in the respective folders (`create_dataset/data/intersentence` and `create_dataset/data/intrasentence`).
Hence, **you can skip to Step 2** if you would work in English or German.
The pickled data files provided version 1.0 and 2.0 are the versions just after the translation without any pre-processing.
Therefore, it's only useful if you want to do a different pre-processing than us.
The main English dataset is in `create_dataset/data/dev.json` and one can translate the dataset and form an appropriate format.
The dataset used is directly copied from [StereoSet](https://github.com/moinnadeem/stereoset/).

To create your own dataset and more details please **follow the instructions in the `create_dataset` folder**

### 2. Acquiring probabilities for candidate words/sentences
After creating the dataset in the pandas format by following Step 1, the probabilities for each candidate can be obtained in this step.  The results of this step (predictions) will be saved in the `results/predictions` folder.
We have provided all the prediction files that we used,
therefore, **you can skip to Step 3** if you would work on the same model with the same dataset.

* **Model Types:** The tests are made with BERT, GPT-2 and T5 based models. Since these comprise all variety of SoTA architectures,
the code should work in all models with optional minor changes.
Direct support for more models and more updates will come according to demand.

* **Complexity**: Utilizing GPU, inferring in batch mode, using multiprocessing is possible in every models by configuring the hyperparameters.
On an average 8 Core CPUs one test is taking around 20 minutes per model on average.
With Tesla V100-SXM2-16GB, using maximum batch size possible, it takes only a few minutes depending on the model.

#### **Fine-tuning Models for Next Sentence Prediction:**
Since GPT and T5 based models were not pre-trained with Next Sentence Prediction objective (for intersentence tests),
we inferred the probabilities using their generative structure. In  addition, we also fine-tuned these models for NSP downstream task to make them comparable with BERT models.
**To Fine-tune models for Next Sentence Prediction (NSP) please follow the instructions in the `code/nsp_training` folder.**
For this, you don't need to run the Step 1.

To obtain the probability predictions for each candidate sentence in the dataset and for more details please **follow the instructions in the `code` folder**.

### 3. Evaluting the Results and Computing Bias
After completing the Step 2 and obtaining the predictions file, one can measure the stereotype score, language modelling score, ICAT score and many more by running the `code/evaluation.py`.

To obtain the final scores and for more details about it please **follow the instructions in the `code` folder.**



## Acknowledgements
This repository makes use of code from the following repositories:
* [StereoSet: Measuring Stereotypical Bias in Pre-trained Language Models](https://github.com/moinnadeem/stereoset)

We acknowledge the support of the following projects and organizations: OpenGPT-X (BMWK 68GX21007C), Alexander Thamm GmbH 
and Ludwig-Maximilians-Universität München.