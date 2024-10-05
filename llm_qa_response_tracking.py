import warnings
warnings.filterwarnings(action = "ignore")

import os
import argparse
import mlflow
from pandas import DataFrame
from time import time

from google_flan_qa_module import Question_Answering_Flan_LLM
from stabilityai_qa_module import QA_StabilityAI_Chat_LLM
from read_data import read_data
import logging

# Create and configure logger
logging.basicConfig(filename = "output_logs.log", format = '%(asctime)s %(message)s', filemode = 'a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

logger.info('\n\n****************** Logging Started ******************')

gt0 = time()

parser = argparse.ArgumentParser(description = 'Accept necessary Command Line Arguments to run this Python Application.')

msg = "The full file path (.csv or .xlsx file) to your custom question answering dataset. Custom Dataset should have two columns : 'inputs' and 'ground_truth'. "
msg += "If no dataset path is entered, default dataset path (./datasets/astronomy_questions_small.xlsx) will be used."
parser.add_argument("-dp", "--dataset_path", help = msg, type = str, default = "./datasets/astronomy_questions_small.xlsx")

msg2 = "Whether to use Google Flan LLMs or Zephyr Chat LLMs. If set to 'true', Google Flan LLM variants will be used. If set to 'false', Zephyr Chat LLM variants will be used.\n"
msg2 += "Dafult choice is 'true'. Models will be downloaded from the HuggingFace Platform."
parser.add_argument("-ugf", "--use_google_flan", help = msg2, type = str, default = 'true')

msg3 = '''Which LLM Model to use. For Google Flan LLMs, any one of the following models can be used : google/flan-t5-small, google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl, google/flan-t5-xxl.\n'''
msg3 += '''For Zephyr Chat LLMs, any one of the following models can be used : stabilityai/stablelm-zephyr-3b, stabilityai/stablelm-2-1_6b-chat, stabilityai/stablelm-2-12b-chat.\n'''
msg3 += "If --use_google_flan argument is set to 'true', select from Google Flan Family of LLMs, else select from Zephyr Chat Family of LLMs.\n"
msg3 += '''Default choice for Flan Model is "google/flan-t5-xl".\nDefault choice for Zephyr Model is "stabilityai/stablelm-zephyr-3b".\n'''
msg3 += "Note : Zephr Chat LLMs are more memory intensive LLMs, compared to Google Flan LLMs. Choose your LLM model accordingly."

parser.add_argument("-lmn", "--llm_model_name", help = msg3, type = str, default = "google/flan-t5-xl")

# Parse the commnad line argumets
args = parser.parse_args()

logger.info(f"  Args Dataset Path : {args.dataset_path}")
logger.info(f"  Args Use Google Flan Models : {args.use_google_flan}")
logger.info(f"  Args LLM Model Name : {args.llm_model_name}")


file_path = args.dataset_path
eval_data = read_data(file_path)

try:
    logger.info("  Creating LLM based QA object and HF Pipeline ...")
    print("\nCreating LLM based QA object ...")
    if args.use_google_flan == 'true':
        llm_qa = Question_Answering_Flan_LLM(args.llm_model_name)
    else:
        llm_qa = QA_StabilityAI_Chat_LLM(args.llm_model_name)
    
    logger.info("  Done.")
    print("Done.")
    
    logger.info(f"  Using Model : {llm_qa.model_name}")
    print(f"\nUsing Model : {llm_qa.model_name}")
    model_name_modified = llm_qa.model_name.replace("/", "_").replace("-", "_")
except Exception as e:
    logger.info("  Error has occured, See below.")
    logger.error(str(e))
    print(str(e))


try:
    hf_pipeline = llm_qa.hf_pipeline

    t0 = time()
    logger.info("  Batch Processing Input Dataset ...")
    print("\nBatch processing Input Dataset ...")
    eval_data_response = llm_qa.batch_process_data(input_df = eval_data)
    t1 = time()
    logger.info(f"  Done. Time Taken : {round(t1-t0, 3)} secs.")
    print(f"Done. Time Taken : {round(t1-t0, 3)} secs.")
except Exception as e:
    logger.info("  Error has occured, See below.")
    logger.error(str(e))
    print(str(e))


tracking_uri = mlflow.get_tracking_uri()
logger.info(f"  Current MLFlow tracking uri: {tracking_uri}")
print(f"\nCurrent MLFlow tracking uri: {tracking_uri}")

mlflow.set_experiment("Evaluate HuggingFace QA Pipeline")

try:
    # Define the signature
    logger.info("  Generating Model Signature ...")
    print("\nGenerating Model Signature ...")
    sample_input = "What are the three primary colors ?"
    params = {"max_new_tokens": 30, "temperature": 0.01}
    signature = mlflow.models.infer_signature(
        sample_input,
        mlflow.transformers.generate_signature_output(hf_pipeline, sample_input, params = params),
        params = params,
    )
    logger.info("  " + str(signature))
    logger.info("  Done.")
    print("Done.")
except Exception as e:
    logger.error("  Error Occurred while Generating Model Signature. See below.")
    logger.error("  " + str(e))
    print("Error Occured while Generating Model Signature. See below.")
    print("\n" + str(e))


# Log the model using mlflowe
with mlflow.start_run(run_name = f"mlflow_run_{model_name_modified}"):
    logger.info("  Extracting Model Info ...")
    print("\nExtracting Model Info ...")
    t0 = time()
    model_info = mlflow.transformers.log_model(
        transformers_model = hf_pipeline,
        artifact_path = model_name_modified,
        registered_model_name = model_name_modified,
        signature = signature,
    )
    
    t1 = time()
    secs = round(t1-t0, 3)
    logger.info(f"  Done. Time taken : {secs} secs.")
    print(f"Done. Time taken : {secs} secs.")

    # Use predefined question-answering metrics to evaluate our model.
    logger.info("  Evaluating Results ...")
    print("\nEvaluating Results ...")
    t0 = time()
    results = mlflow.evaluate(
        model_info.model_uri,
        eval_data_response,
        targets = "ground_truth",
        predictions = "llm_response",
        # model_type = "question-answering",  ## comment it out to use custom metrics in extra_metrics parameter
        extra_metrics = [mlflow.metrics.latency(), mlflow.metrics.toxicity(), 
                         mlflow.metrics.ari_grade_level(), 
                         mlflow.metrics.flesch_kincaid_grade_level()],
    )

    t1 = time()
    secs = round(t1-t0, 3)
    logger.info(f"  Done. Time taken : {secs} secs.")
    print(f"Done. Time taken : {secs} secs.")

    print(f"\nSee aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    temp = results.tables["eval_results_table"]
    eval_table = DataFrame()
    eval_table["inputs"] = eval_data_response["inputs"]
    eval_table["ground_truth"] = eval_data_response["ground_truth"]
    eval_table["llm_response"] = eval_data_response["llm_response"]
    eval_table["latency_score"] = temp["latency"]
    eval_table["toxicity_score"] = temp["toxicity/v1/score"]
    eval_table["ari_grade_level_score"] = temp["ari_grade_level/v1/score"]
    eval_table["flesch_kincaid_grade_level_score"] = temp["flesch_kincaid_grade_level/v1/score"]

    print(f"See evaluation table below: \n{eval_table}")
    
    # Log the Metrics CSV File as Artifact
    # csv_path = "./eval_table_metrics.csv"
    csv_path = f"./metric_files/{model_name_modified}_eval_results.csv"
    logger.info(f" The Results CSV File getting Tracked is : {csv_path}")
    print(f"\nThe Results CSV File getting Tracked is : {csv_path}")

    eval_table.to_csv(csv_path, header = True, index = False, sep = ",", mode = "w+")
    mlflow.log_artifact(csv_path)


# Release Memory
eval_data, eval_data_response, temp = [], [], []

gt1 = time()
g_secs = round(gt1-gt0, 3)
logger.info(f"  QA Evaluation Process Finished. Time taken : {g_secs} secs.")
print(f"\nQA Evaluation Process Finished. Time taken : {g_secs} secs.")

with mlflow.start_run(run_name = f"extra_artifacts_{model_name_modified}"):
    try:
        os.system("pip freeze > requirements.txt")
    except:
        try:
            os.system("pip3 freeze > requirements.txt")
        except Exception as e:
            logger.error("  Some error has occured while trying to produce the requirements.txt file. See below.")
            logger.error("  " + str(e))
            print("\nSome error has occured while trying to produce the requirements.txt file. See below.")
            print(str(e))
    
    mlflow.log_artifact("./output_logs.log")
    mlflow.log_artifact("./requirements.txt")

