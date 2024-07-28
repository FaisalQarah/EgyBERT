###### pretraining the model

from datasets import load_from_disk

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

from datasets import load_dataset
import glob
import tokenizers
from transformers import Trainer, TrainingArguments, LineByLineTextDataset, BertModel
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoTokenizer

dataset = load_from_disk("tokenized_dataset_EgyBERT/")

# max_seq_length = 128
tokenizer = AutoTokenizer.from_pretrained('EgyBERT/')
config = BertConfig( vocab_size = 75000, 
                    hidden_size = 768, 
                    num_hidden_layers = 12,
                    num_attention_heads = 12,
                    max_position_embeddings = 512)

model = BertForMaskedLM(config)
print(model.num_parameters())


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=0.15)
epochs = 24.52
save_steps = 50_000 # 20_000 #save checkpoint every 10000 steps
batch_size = 64

training_args = TrainingArguments(
    output_dir = 'EgyBERT/',
    overwrite_output_dir=True,
    num_train_epochs = epochs,
    per_device_train_batch_size = batch_size,
    save_steps = save_steps,
    save_total_limit = 3, #only save the last 5 checkpoints
    fp16=True,
    # tf32 = True,
    learning_rate = 5e-5,  # 5e-5 is the default
    logging_steps = 25_000,
    gradient_accumulation_steps=2,
    # gradient_checkpointing=True,
    #lr_scheduler_type='constant',
    max_grad_norm=1.5, #gradient clipping
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator=data_collator,
    train_dataset=dataset

)


#trainer.train(resume_from_checkpoint=True)
trainer.train()
trainer.save_model('EgyBERT/')
