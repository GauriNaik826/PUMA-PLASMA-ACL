import json
import argparse 
from transformers.file_utils import PushToHubMixin
from transformers import BartTokenizer, BartForConditionalGeneration,GPT2LMHeadModel,AutoModelForSeq2SeqLM,AutoTokenizer,BertTokenizer,BertModel
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
import sys
sys.path.insert(0, './') 
from src.FLANT5.prefix_energy.prefix_energy_train_dataloader import * 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import numpy as np
import os
import torch
from scipy.spatial.distance import cosine
from bert_score import score
import math
from rouge import Rouge
import numpy as np
import warnings
warnings.filterwarnings("ignore")
device = 'cuda'
from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
import numpy as np
import pandas as pd


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)
ckpt_path = f"./classifier/checkpoint_classifier"
if os.path.exists(ckpt_path):
        print("Loading the trained checkpoint...")
        ckpt = torch.load(ckpt_path)
        roberta_model.load_state_dict(ckpt['model_state_dict'])
        print("The inference will start with the specified checkpoint.")



    
def get_bert_embedding(text):
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()


def calculate_rouge_score_for_each_phrase(predictions, references):
        rouge = Rouge()
        rouge_l_f1_scores = []

        for prediction, reference in zip(predictions, references):
            scores = rouge.get_scores(prediction.lower(), reference.lower())[0]
            rouge_l_f1 = scores["rouge-1"]["f"]
            rouge_l_f1_scores.append(rouge_l_f1)

        return rouge_l_f1_scores

def score_all_phrases(summary, phrases):
        start_of_summary = ' '.join(summary.split()[:4])

    
        predictions = [start_of_summary] * len(phrases)
        references = phrases

        rouge_l_f1_results = calculate_rouge_score_for_each_phrase(predictions, references)

        phrase_scores = dict(zip(phrases, rouge_l_f1_results))

        return phrase_scores

def Ep(generated_summary):

        inputs = roberta_tokenizer(generated_summary, padding=True, truncation=True, return_tensors="pt").to(device)

    
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = outputs.logits.argmax(dim=-1)

        class_labels = {0: "EXPERIENCE", 1: "SUGGESTION", 2: "INFORMATION", 3: "CAUSE", 4: "QUESTION"}
        predicted_label = class_labels[predictions[0].item()]

        final ={}
        for i in range(0,5):
            final[class_labels[i]] = probabilities[0][i].cpu().numpy().item()

        return final


def Es(generated_summary):
        phrases = [
            "In user's experience…",
            "It is suggested",
            "For information purposes",
            "Some of the causes",
            "It is inquired"
        ]
        
        phrase_scores = score_all_phrases(generated_summary, phrases)
        return phrase_scores

def Et(generated_summary):
        l_sugg =  ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding","Prescriptive"]
        l_exp = ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
        l_info =  ["Clinical", "Scientific","Informative", "Educational","Factual", "Informing","Academic","Analytical"]
        l_cause =  ["Diagnostic", "Explanatory", "Causal","Due to", "Resulting from", "Attributable to" ]
        l_qs =  ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry", "Problem-Solving Deliberation"]


        summary_embedding = get_bert_embedding(generated_summary)

        cosine_similarities = {}

        for label, word_list in zip(['sugg', 'exp', 'info', 'cause', 'qs'], [l_sugg, l_exp, l_info, l_cause, l_qs]):
            combined_text = ' '.join(word_list)
            word_embedding = get_bert_embedding(combined_text)
            similarity = 1 - cosine(summary_embedding.cpu().detach().numpy(), word_embedding.cpu().detach().numpy())
            cosine_similarities[label] = similarity

        return cosine_similarities

def compute_custom_loss(model, input_text, input_attention, perspective):
        model.eval()
        outputs = model.generate(input_ids=input_text,attention_mask=input_attention,num_beams=5, max_new_tokens=100,temperature=0.9)
        generated_summary = tokenizer.decode(outputs[0])
        if len(generated_summary) <= 0:
            generated_summary = 'None'
       
        Ep_dict = Ep(generated_summary)
        Es_dict = Es(generated_summary)
        Et_dict = Et(generated_summary)


        alpha = 0.7  
        beta = 0.3   
        gamma = 0.5  

        E_X = {
            "EXPERIENCE": alpha * Ep_dict["EXPERIENCE"] + beta * Es_dict["In user's experience…"] + gamma * Et_dict['exp'],
            "SUGGESTION": alpha * Ep_dict["SUGGESTION"] + beta * Es_dict["It is suggested"] + gamma * Et_dict['sugg'],
            "INFORMATION": alpha * Ep_dict["INFORMATION"] + beta * Es_dict["For information purposes"] + gamma * Et_dict['info'],
            "CAUSE": alpha * Ep_dict["CAUSE"] + beta * Es_dict["Some of the causes"] + gamma * Et_dict['cause'],
            "QUESTION": alpha * Ep_dict["QUESTION"] + beta * Es_dict["It is inquired"] + gamma * Et_dict['qs']
        }

        # Compute the exponential of E(X) for normalization
        exp_E_X = {k: math.exp(-1/(v)) for k, v in E_X.items()}
        
        # Compute the sum of the exponentials for normalization
        Z = sum(exp_E_X.values())

        # Calculate P(X) for each perspective
        P_X = {k: v / Z for k, v in exp_E_X.items()}
        

        Y = {"EXPERIENCE": 0, "SUGGESTION": 0, "INFORMATION": 0, "CAUSE": 0, "QUESTION": 0}
        Y[perspective[0]] = 1

        P_X_tensor = torch.tensor([P_X[k] for k in P_X])
        Y_tensor = torch.tensor([Y[k] for k in Y])

        loss = -torch.sum(Y_tensor * torch.log(P_X_tensor))
        return loss 


            


def validation(valid_dataloader, model, VALID_BATCH_SIZE, optimizer, scheduler):
        
            print("Validation processing...")
            model.eval()    
            valid_losses = []
            gen = []
            actual =[]
            with torch.no_grad():
                for i,batch in enumerate(tqdm(valid_dataloader)):
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['input_ids'].to(device)
                    
                    output = model(input_ids= input_ids,attention_mask=attention_mask,labels=labels)
                
                    custom_loss = compute_custom_loss(model,input_ids,attention_mask, batch["perspective"])
                    loss = output.loss + custom_loss

                   

                    outputs = model.generate(input_ids=input_ids,attention_mask=attention_mask,num_beams=5, max_new_tokens=100,temperature=0.9)
                    generated_summary = tokenizer.decode(outputs[0])
                    gen.append(generated_summary)
                    actual.append(batch['Summary'])
                  
                    
                    print(f"_________________ValidBatch: {i}/{len(valid_dataloader)} || ValidLoss: {loss}_____________________")
                    valid_losses.append(loss.item()) 
                    
            valid_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.0  
            return valid_loss 

if __name__=="__main__":

    ##########################################################################
    # Prepare Parser
    ##########################################################################
        list_loss_train = []
        list_loss_valid = []
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_file', required=True)
        parser.add_argument('--valid_file', type=str, required=True)
        parser.add_argument('--batch_size_train', type=int, required=True)
        parser.add_argument('--batch_size_valid', type=int, required=True)
        parser.add_argument('--model_file', type=str, required=True)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--warmup_steps", type=int, default=16000)
        parser.add_argument("--num_epochs", type=int, default=None,
                        help="how many training epochs")
        parser.add_argument("--ckpt_dir", type=str, default=None)
        parser.add_argument("--ckpt_name", type=str, default=None)
        
        args = parser.parse_args()
        

        TRAIN_BATCH_SIZE = args.batch_size_train
        with open(args.train_file, 'r') as json_file:
            train_data = json.load(json_file)
        VALID_BATCH_SIZE = args.batch_size_valid
        with open(args.valid_file, 'r') as json_file:
            valid_data = json.load(json_file)
        LR = args.learning_rate
        WARMUP_STEPS = args.warmup_steps
        EPOCHS = args.num_epochs
        best_loss = sys.float_info.max 
        last_epoch = 0

       
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_file)
        tokenizer = AutoTokenizer.from_pretrained(args.model_file)
       


        peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        num_virtual_tokens=8, 
        token_dim=1024 
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        model = peft_model

                
        train_dataset = CustomDataset(train_data,tokenizer)
        eval_dataset = CustomDataset(valid_data,tokenizer)
        train_dataloader, eval_dataloader = create_dataloader(train_dataset, eval_dataset,  VALID_BATCH_SIZE, TRAIN_BATCH_SIZE)
        

        # Define optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_dataloader) * EPOCHS)

        if args.ckpt_name is not None:
                ckpt_path = f"{args.ckpt_dir}/{args.ckpt_name}.ckpt"
                print("_________ckpt_path___________",ckpt_path)
                if os.path.exists(ckpt_path):
                
                    print("Loading the trained checkpoint...")
                    ckpt = torch.load(ckpt_path)
                    model.load_state_dict(ckpt['model_state_dict'])
        
                    print(f"The training restarts with the specified checkpoint: {args.ckpt_name}.ckpt.")
                    optimizer.load_state_dict(ckpt['optim_state_dict'])
                    scheduler.load_state_dict(ckpt['sched_state_dict'])
                    loss = ckpt['loss']
                    last_epoch = ckpt['epoch']   
                else:
                    print(f"Cannot find the specified checkpoint {ckpt_path}.")
    
        start_epoch = last_epoch+1
        num_batches = len(train_dataloader)

       
        model.to(device)
        # Fine-tuning loop
        for epoch in range(start_epoch,start_epoch+ EPOCHS):
            model.train()
            print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            train_losses = []
            for i,batch in enumerate(tqdm(train_dataloader)):
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels =  batch["labels"].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                custom_loss = compute_custom_loss(model,input_ids,attention_mask, batch["perspective"])
             
                loss = outputs.loss + custom_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.detach())

            train_losses = [loss.item() for loss in train_losses] 
            train_loss = np.mean(train_losses)
            print(f"Train loss: {train_loss} for epoch : {epoch}")
            list_loss_train.append(train_loss)
            model.save_pretrained(f"{args.ckpt_dir}/best_ckpt_epoch={epoch}")
           
            valid_loss = validation(eval_dataloader, model, VALID_BATCH_SIZE, optimizer, scheduler)

            if valid_loss < best_loss:
                    best_loss = valid_loss
                    state_dict = {
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'sched_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'epoch': last_epoch
                    }
                
                    model.save_pretrained(f"{args.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(best_loss, 4)}")
                   
            
            

        








