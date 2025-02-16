import torch
from torch.utils.data import Dataset,DataLoader
from transformers import RobertaTokenizer

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_map = {"EXPERIENCE": 0, "SUGGESTION": 1, "INFORMATION": 2, "CAUSE": 3, "QUESTION": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]['Summary']
        label = self.data[idx]['Perspective']

        # Encode label to integer
        label_encoded = self.label_map[label]


        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Return input_ids, attention_mask, and label
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(label_encoded)  # Return integer label directly
        }

def create_dataloader(train_dataset,valid_dataset, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE ):
    
    train_dataloader= DataLoader(dataset = train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle= True)
    valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = VALID_BATCH_SIZE, shuffle= True)
    
    return train_dataloader , valid_dataloader

def test_create_dataloader(test_dataset, TEST_BATCH_SIZE ):
    
    test_dataloader= DataLoader(dataset = test_dataset, batch_size = TEST_BATCH_SIZE, shuffle= True)
     
    return test_dataloader



    
