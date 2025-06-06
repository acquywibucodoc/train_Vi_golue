# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Trainer, pipeline, TrainingArguments
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import json

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_path = r"E:\data luatvietnam.vn\checkpoint-116000\checkpoint-116000"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

# Set up the chat format
#model, tokenizer = setup_chat_format(model=None, tokenizer=tokenizer)

print("✅ Loaded the mf qwen model")

def load_json_dataset(path, val_size=0.1, seed=42):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data, val_data = train_test_split(data, test_size=val_size, random_state=seed)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset

# Usage
train_path = r"qqp\train.json"
train_dataset, val_dataset = load_json_dataset(train_path, val_size=0.1)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

def formatting_prompt_func_QQP(example):
    return (
        "<Instruct>Do these two questions convey the same meaning? (answer duplicate or not_duplicate)</Instruct>\n"
        "<Format>Question pair format</Format>\n"
        f"<Question>Question 1: {example['question1']}\nQuestion 2: {example['question2']}</Question>\n"
        f"<Answer>{example['label']}</Answer>"
    )

training_args = TrainingArguments(
    output_dir="Qwenv2.5_QQP_SFT_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,  
    load_best_model_at_end=True,
    push_to_hub=False,  
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompt_func_QQP,
    )

trainer.train()

metrics = trainer.evaluate()

print(metrics)