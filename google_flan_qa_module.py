import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Create and configure logger
logging.basicConfig(filename = "output_logs.log", format = '%(asctime)s %(message)s', filemode = 'a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Question_Answering_Flan_LLM():

    def __init__(self, model_name: str = "google/flan-t5-xl", generation_params: dict = {}):
        self.model_name = model_name
        self.generation_params = generation_params
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = self.model_name)
        self.hf_pipeline = pipeline(task = "text2text-generation", model = self.model_name, 
                                    tokenizer = self.tokenizer, device = "cpu")
        
        flan_models = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", 
                       "google/flan-t5-xl", "google/flan-t5-xxl"]
        
        if self.model_name not in flan_models:
            text = f"Allowed models : {flan_models}.\nReceived model name : {self.model_name}.\nConsider creating your own custom class for using LLMs of your own choice."
            logger.error("  " + text)
            raise ValueError(text)
        
        if self.generation_params == {}:
            self.generation_params = {"max_new_tokens": 512, "temperature": 0.03, "repetition_penalty": 1.5, 
                                      "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id, 
                                      "bos_token_id": self.tokenizer.bos_token_id}
        else:
            if "eos_token_id" not in self.generation_params.keys():
                self.generation_params["eos_token_id"] = self.tokenizer.eos_token_id
            if "pad_token_id" not in self.generation_params.keys():
                self.generation_params["pad_token_id"] = self.tokenizer.pad_token_id
            if "bos_token_id" not in self.generation_params.keys():
                self.generation_params["bos_token_id"] = self.tokenizer.bos_token_id
            
            if "temperature" not in self.generation_params.keys():
                self.generation_params["temperature"] == 0.03
            
            if self.generation_params["temperature"] > 0.0:
                self.generation_params["do_sample"] = True
            else:
                self.generation_params["do_sample"] = False
        
        logger.info(f"Using {self.model_name} LLM Model for Question Answering Task.")
        print(f"\nUsing {self.model_name} LLM Model for Question Answering Task.")

        logger.info(f"Generation Params : {self.generation_params}")
        print(f"\nGeneration Params : {self.generation_params}")


    def generate_response(self, input_question: str, prompt: str = None):
        self.input_question = str(input_question)
        self.prompt = prompt
        if self.prompt is None:
            self.prompt = "Answer the following Question : {}"
        else:
            pass
        
        self.prompt = str(self.prompt).format(self.input_question)
        input_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids
        
        outputs = self.model.generate(input_ids, **self.generation_params)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        return str(answer)
    

    def batch_process_data(self, input_df : pd.DataFrame, prompt: str = None):
        answers = []

        for i in range(len(input_df)):
            question = str(input_df["inputs"].iloc[i])
            answer = self.generate_response(input_question = question, prompt = str(prompt))
            answers.append(answer)
        
        input_df["llm_response"] = answers
        return input_df
