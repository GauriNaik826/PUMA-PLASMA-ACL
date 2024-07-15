import torch
from torch.utils.data import Dataset, DataLoader
import json  
from transformers import BartTokenizer, BartForConditionalGeneration,AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer,max_length=1024):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        answers = self.data[idx]['answers']
        target_text = self.data[idx]['Summary']
        defn = ""
        start_with = ""
        tone_attribute = ""
        l = []
        target_text = ''
        if self.data[idx]['Perspective'].strip() ==  "SUGGESTION":
            defn = "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues."
            start_with =  "It is suggested"
            tone_attribute = "Advisory, Recommending"
            l =  ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding","Prescriptive"]
            if len(set(self.data[idx]['Summary'].split(" ")[:5]).intersection(set(start_with.split()))) > 3  :
                target_text = self.data[idx]['Summary']
            else:
                target_text = start_with + " " +self.data[idx]['Summary']
        if self.data[idx]['Perspective'].strip() == "INFORMATION":
            defn = "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis."
            start_with =  "For information purposes"
            tone_attribute = "Informative, Educational"
            l =  ["Clinical", "Scientific","Informative", "Educational","Factual", "Informing","Academic","Analytical"]
            if len(set(self.data[idx]['Summary'].split(" ")[:5]).intersection(set(start_with.split()))) >= 2:
                target_text = self.data[idx]['Summary']
            else:
                target_text = start_with + " " +self.data[idx]['Summary']
        if self.data[idx]['Perspective'].strip() == "EXPERIENCE":
            defn = "Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, medication usage, and coping strategies"
            start_with =  "In user's experience"
            tone_attribute = "Personal, Narrative"
            l =  ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"]
            if len(set(self.data[idx]['Summary'].split(" ")[:5]).intersection(set(start_with.split()))) >= 2:
                target_text = self.data[idx]['Summary']
            else:
                target_text = start_with + " " +self.data[idx]['Summary']
        if self.data[idx]['Perspective'].strip() == "CAUSE":
            defn = "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease"
            start_with =  "Some of the causes"
            tone_attribute = "Explanatory, Causal"
            l =  ["Diagnostic", "Explanatory", "Causal","Due to", "Resulting from", "Attributable to" ]
            if len(set(self.data[idx]['Summary'].split(" ")[:5]).intersection(set(start_with.split()))) >= 2:
                target_text = self.data[idx]['Summary']
            else:
                target_text = start_with + " " +self.data[idx]['Summary']
        if self.data[idx]['Perspective'].strip() == "QUESTION":
            defn = "Defined as inquiry made for deeper understanding."
            start_with =  "It is inquired"
            tone_attribute = "Seeking Understanding"
            l =  ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry", "Problem-Solving Deliberation"]
            if len(set(self.data[idx]['Summary'].split(" ")[:5]).intersection(set(start_with.split()))) >= 2:
                target_text = self.data[idx]['Summary']
            else:
                target_text = start_with + " " +self.data[idx]['Summary']
        
        non_empty_sentences = ' '.join([sentence.replace('\n', '') for sentence in answers])
       
        task_prefix = "Adhering to the condition of 'begin summary with' and 'tone of summary' and summarize according to "+self.data[idx]['Perspective'].strip()+ "and start the summary with '"+ start_with.strip()+ "'. Maintain summary tone as " + tone_attribute.strip()+ ". Definition of perspective: "+ defn.strip().lower() + " Content to summarize: "+ non_empty_sentences +" Question: "+ self.data[idx]['question'].strip()+"."
        
        inputs = self.tokenizer(task_prefix, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        labels = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "perspective": self.data[idx]['Perspective'],
            "Summary": self.data[idx]['Summary']
           
        }
def create_dataloader(train_dataset,valid_dataset, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE ):
    
    train_dataloader= DataLoader(dataset = train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle= True)
    valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = VALID_BATCH_SIZE, shuffle=True)
    
    return train_dataloader , valid_dataloader

def test_create_dataloader(test_dataset, TEST_BATCH_SIZE ):
    
    test_dataloader= DataLoader(dataset = test_dataset, batch_size = TEST_BATCH_SIZE, shuffle= False)
     
    return test_dataloader





    
      