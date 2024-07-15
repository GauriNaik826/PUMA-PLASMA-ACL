import json
import argparse 
from transformers.file_utils import PushToHubMixin
from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel,AutoModelForSeq2SeqLM,AutoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType,PeftModel
import sys
sys.path.insert(0, './') 
from src.train_dataloader import * 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
device = 'cuda'
if __name__=="__main__":

##########################################################################
# Prepare Parser
##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--batch_size_test', type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, default=None)
    
    args = parser.parse_args()
    
    TEST_BATCH_SIZE = args.batch_size_test
    with open(args.test_file, 'r') as json_file:
        test_data = json.load(json_file)
    EPOCHS = args.num_epochs
    
    foundation_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_file).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_file)

    peft_model_path = f"{args.ckpt_dir}/{args.ckpt_name}"
    
    loaded_model = PeftModel.from_pretrained(
    foundation_model,  # The base model to be used for prefix tuning
    peft_model_path,   # The path where the trained Peft model is saved
    is_trainable=False  # Indicates that the loaded model should not be trainable
    ).to(device)
    
    test_dataset = CustomDataset(test_data,tokenizer)
    test_dataloader = test_create_dataloader(test_dataset, TEST_BATCH_SIZE)

            
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            input_text = batch["input_ids"].to(device)
            input_attention = batch["attention_mask"].to(device)
            outputs =  loaded_model.generate(input_ids=input_text,attention_mask=input_attention,num_beams=5, max_new_tokens=500,temperature=0.9, repetition_penalty=1.2)
         
            output_text = tokenizer.decode(outputs[0]).replace('<pad>','').replace('</s>','').strip(" ")
            data = {'PERSPECTIVE':test_data[step]['Perspective'],'PREDICTED': [output_text], 'ACTUAL OUTPUT':test_data[step]['Summary'],'INPUT':[test_data[step]['answers']]}
            print(data)
            df= pd.DataFrame(data)
            df.to_csv('./generated/generated_result.csv', mode='a', index=False, header=False)

        









