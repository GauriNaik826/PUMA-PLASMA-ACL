import json
import argparse 
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import sys
sys.path.insert(0, './') 
from classifier.classifier_dataloader import *
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
def validation(valid_dataloader, model, VALID_BATCH_SIZE, optimizer, scheduler):
       
        print("Validation processing...")
        model.eval()    
        valid_losses = []
      
        with torch.no_grad():
            for i,batch in enumerate(tqdm(valid_dataloader)):
                
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['label'].to('cuda')
                
                output = model(input_ids= input_ids,attention_mask=attention_mask,labels=labels)
             
                loss = output.loss
               
                
                #print(f"_________________ValidBatch: {i}/{len(valid_dataloader)} || ValidLoss: {loss}_____________________")
                valid_losses.append(loss.item())  
            
        valid_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.0  # Calculate mean of valid_losses
        return valid_loss 

if __name__=="__main__":
   
    TRAIN_BATCH_SIZE = 8
    with open('/home/gaurin/Summ_B/data/PREPROCESSED/preprocessed_train_data.json', 'r') as json_file:
        train_data = json.load(json_file)

    VALID_BATCH_SIZE = 4
    with open('/home/gaurin/Summ_B/data/PREPROCESSED/preprocessed_valid_data.json', 'r') as json_file:
        valid_data = json.load(json_file)
    
    LEARNING_RATE = 1e-05
    LR =  1e-05
    WARMUP_STEPS = 4000
    EPOCHS = 5
    # NUM_OPTIM_STEPS = args.num_optim_steps
    best_loss = sys.float_info.max 
    print("best_loss",best_loss)
    last_epoch = 0

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)  # Change num_labels to match your number of classes
            
    train_dataset = CustomDataset(train_data,tokenizer)
    eval_dataset = CustomDataset(valid_data,tokenizer)
    train_dataloader, eval_dataloader = create_dataloader(train_dataset, eval_dataset,  VALID_BATCH_SIZE, TRAIN_BATCH_SIZE)
    

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_dataloader) * EPOCHS)

    start_epoch = last_epoch+1
    num_batches = len(train_dataloader)

   
    model.to('cuda')
    # Fine-tuning loop
    for epoch in range(start_epoch,start_epoch+ EPOCHS):
        model.train()
        print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
        train_losses = []
        for i,batch in enumerate(tqdm(train_dataloader)):
            
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.detach())
         
        train_losses = [loss.item() for loss in train_losses] 
        train_loss = np.mean(train_losses)
        print(f"Train loss: {train_loss} for epoch : {epoch}")
        
        #Save the checkpoint for the batch with best loss after comparing with validation
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
               
                torch.save(state_dict, f"./classifier/classifier_dataloader.py/best_ckpt_epoch={epoch}_valid_loss={round(best_loss, 4)}.ckpt")
                print("*"*10 + "Current best checkpoint is saved." + "*"*10)
              
        

    









