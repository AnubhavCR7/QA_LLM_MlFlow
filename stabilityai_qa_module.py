import logging
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# Create and configure logger
logging.basicConfig(filename = "output_logs.log", format = '%(asctime)s %(message)s', filemode = 'a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class QA_StabilityAI_Chat_LLM():

    def __init__(self, model_name: str = "stabilityai/stablelm-zephyr-3b", generation_params: dict = {}):
        self.model_name = model_name
        self.generation_params = generation_params
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.model_name, padding = True, truncation = True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = self.model_name)
        self.hf_pipeline = pipeline(task = "text-generation", model = self.model_name, 
                                    tokenizer = self.tokenizer, return_full_text = False)
        
        stabilityai_chat_models = ["stabilityai/stablelm-zephyr-3b", "stabilityai/stablelm-2-1_6b-chat",
                                    "stabilityai/stablelm-2-12b-chat"]
        if self.model_name not in stabilityai_chat_models:
            text = f"Allowed models : {stabilityai_chat_models}.\nReceived model name : {self.model_name}.\nConsider creating your own custom class for using Chat LLMs of your own choice."
            logger.error("  " + text)
            raise ValueError(text)
        
        if self.generation_params == {}:
            self.generation_params = {"max_new_tokens": 360, "temperature": 0.01, "do_sample": True, "repetition_penalty": 1.5, 
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
                self.generation_params["temperature"] == 0.01
            
            if self.generation_params["temperature"] > 0.0:
                self.generation_params["do_sample"] = True
            else:
                self.generation_params["do_sample"] = False
        
        logger.info(f"Using {self.model_name} Chat LLM Model for Question Answering Task.")
        print(f"\nUsing {self.model_name} Chat LLM Model for Question Answering Task.")

        logger.info(f"Generation Params : {self.generation_params}")
        print(f"\nGeneration Params : {self.generation_params}")
    

    def generate_response(self, input_question: str, prompt: str = None):
        self.input_question = str(input_question)
        self.prompt = prompt
        if self.prompt is None:
            # We use the tokenizer's chat template to format each message
            self.prompt = "You are a helpful AI Assistant whose task is to answer the questions asked by the user as truthfully and honestly as possible. \
Stick to the facts while answering the user question. DO NOT USE ANY ABUSIVE OR VULGAR LANGUAGE while answering the question. Answer the questions in a formal and professional tone.\
If you don't know the answer to any question, ALWAYS RESPOND WITH 'Sorry! I do not have any answer to this question.'"
        else:
            pass
        
        messages = [
                        {"role": "system", "content": self.prompt}, 
                        {"role": "user", "content": self.input_question},
                    ]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        input_ids = self.tokenizer(input_text, return_tensors = "pt").input_ids
        
        outputs = self.model.generate(input_ids, **self.generation_params)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        answer = str(answer)
        answer = answer.split("<|assistant|>")[-1].strip("\n")
        return answer
    

    def batch_process_data(self, input_df : pd.DataFrame, prompt: str = None):
        answers = []

        for i in range(len(input_df)):
            question = str(input_df["inputs"].iloc[i])
            answer = self.generate_response(input_question = question, prompt = str(prompt))
            answers.append(answer)
        
        input_df["llm_response"] = answers
        return input_df

