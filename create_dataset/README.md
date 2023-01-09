After running below codes, your new pandas dataframe will be pickled into the `create_dataset/data/intersentence` and `create_dataset/data/intrasentence` folders accordingly.
### To use it in English
1. `cd create_dataset`
2. `python3 data_creator.py`
### To Create Datasets in Different Languages
The translations are done with AWS Translate (https://aws.amazon.com/translate/) and 2 Million characters,
which is extremely high, is free. Hence, the aws access key id and secret access key should be given to the program.
The target language for the translations must also be specified.
There are [75 languages](https://aws.amazon.com/translate/faqs/) supported in AWS Translate.
1. `cd create_dataset`
2. `python3 data_creator.py --aws-access-key-id <your_key_id> --aws-secret-access-key <your_secret_key> --language <lang>`

####Note
In case of a run in German language, german_translation_fixes.py file will be called and German-specific fixes will
be done automatically. You can add different rules there or create a new file for a new language, however,
this is optional, it is observed that the results without any fix is already satifying.