import os

from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)
# Predefined choices
pretrained_model_choices = ['BERT', 'Multilingual BERT', "GPT-2",
                            'Multilingual GPT-2', 'T5', "Multilingual T5"]
tokenizer_choices = ["BertTokenizer", "GPT2Tokenizer", "T5Tokenizer", "MT5Tokenizer", "AutoTokenizer",
                     "GPT2TokenizerFast"]

def initialize_params(pretrained_model, language):
    pretrained_class = "bert-base-cased"
    intrasentence_model = "BertLM"
    intersentence_model = "BertNextSentence"
    if pretrained_model == "BERT":
        if language in "de":
            pretrained_class = "dbmdz_bert-base-german-cased"
        elif language == "es":
            pretrained_class = "dccuchile_bert-base-spanish-wwm-cased"
        elif language == "fr":
            pretrained_class = "flaubert_flaubert_base_cased"
        elif language == "tr":
            pretrained_class = "dbmdz_bert-base-turkish-cased"

    elif pretrained_model == "Multilingual BERT":
        pretrained_class = "bert-base-multilingual-cased"

    elif pretrained_model == "GPT-2":
        intrasentence_model = "GPT2LM"
        intersentence_model = "GPT2LM_d"
        if language == "en":
            pretrained_class = "gpt2"
        elif language in "de":
            pretrained_class = "dbmdz_german-gpt2"
        elif language == "es":
            pretrained_class = "PlanTL-GOB-ES_gpt2-base-bne"
        elif language == "fr":
            pretrained_class = "asi_gpt-fr-cased-small"
        else:
            pretrained_class = "redrussianarmy_gpt2-turkish-cased"

    elif pretrained_model == "Multilingual GPT-2":
        pretrained_class = "THUMT_mGPT"
        intrasentence_model = "GPT2LM"
        intersentence_model = "GPT2LM_d"

    elif pretrained_model == "T5":
        intrasentence_model = "T5LM"
        intersentence_model = "ModelNSP"
        if language == "en":
            pretrained_class = "t5-base"
        elif language in "de":
            pretrained_class = "GermanT5_t5-efficient-gc4-german-base-nl36"
        elif language == "es":
            pretrained_class = "flax-community_spanish-t5-small"
        elif language == "fr":
            pretrained_class = "plguillou_t5-base-fr-sum-cnndm"
        else:
            print("Turkish has no T5")
            pretrained_class = "!"
    elif pretrained_model == "Multilingual T5":
        pretrained_class = "google_mt5-base"
        intrasentence_model = "mT5LM"
        intersentence_model = "ModelNSP"


    intrasentence_df_file = "../create_dataset/data/intrasentence/df_intrasentence_" + language + ".pkl"
    intersentence_df_file = "../create_dataset/data/intersentence/df_intersentence_" + language + ".pkl"

    predictions_file = "../results/predictions/" + pretrained_class + "_" + intrasentence_model + "_" \
                       + intersentence_model + "_" + language + ".json"

    return pretrained_class.replace('_', '/', 1), intrasentence_model, intersentence_model.split("_")[0], intrasentence_df_file, intersentence_df_file, \
        predictions_file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pretrained_model = request.form.get('pretrained_model')
        device = request.form.get('device')
        test_type = request.form.get('test_type')
        nsp_model_path = request.form.get('nsp_model_path')
        tokenizer_name = request.form.get('tokenizer_name')
        batch_size = request.form.get('batch_size')
        num_workers = request.form.get('num_workers')
        case_no = request.form.get('case_no')
        output_file = request.form.get('output_file')
        skip_inference = 'skip_inference' in request.form
        language = request.form.get('language')

        pretrained_class, intrasentence_model, intersentence_model, intrasentence_df_file, intersentence_df_file, \
            predictions_file = initialize_params(pretrained_model, language)

        # Prepare the command to run the Python application
        command = [
            "python", "../code/main.py",
            "--pretrained-class", pretrained_class,
            "--intrasentence-df-file", intrasentence_df_file,
            "--intersentence-df-file", intersentence_df_file,
            "--intrasentence-model", intrasentence_model,
            "--intersentence-model", intersentence_model,
            "--tokenizer-name", tokenizer_name,
            "--batch-size", batch_size,
            "--num-workers", num_workers,
            "--case-no", case_no,
            "--device", device
        ]

        if predictions_file:
            command.extend(["--predictions-file", predictions_file])
        if output_file:
            command.extend(["--output-file", output_file])
        if nsp_model_path:
            command.extend(["--nsp-model-path", nsp_model_path])
        if test_type == "intersentence":
            command.append("--skip-intrasentence")
        if test_type == "intrasentence":
            command.append("--skip-intersentence")
        if skip_inference:
            command.append("--skip-inference")

        # Run the command
        print(command)
        try:
            subprocess.run(command, text=True)
        except:
            print("error")
        image_exists = os.path.isfile('static/scatter_plot.png')
        return render_template('index.html', pretrained_model_choices=pretrained_model_choices,
                               tokenizer_choices=tokenizer_choices, image_exists=image_exists,
                               pretrained_model=pretrained_model, device=device, test_type=test_type,
                               nsp_model_path=nsp_model_path, tokenizer_name=tokenizer_name,
                               batch_size=batch_size, num_workers=num_workers, case_no=case_no,
                               output_file=output_file, skip_inference=skip_inference, language=language)
    image_exists = os.path.isfile('static/scatter_plot.png')
    print(image_exists)
    return render_template('index.html', pretrained_model_choices=pretrained_model_choices,
                           tokenizer_choices=tokenizer_choices, image_exists=image_exists)

if __name__ == '__main__':
    app.run(debug=True)
