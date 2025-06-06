# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Trainer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_path = "./checkpoint-116000"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

# Set up the chat format
#model, tokenizer = setup_chat_format(model=None, tokenizer=tokenizer)

def load_json_dataset(train_path, val_path):
    dataset = load_dataset('json', data_files={
        'train': train_path,
        'validation': val_path
    })
    return dataset['train'], dataset['validation']

train_path = r"rte\train.json"
val_path = r"rte\validation.json"

train_dataset, val_dataset = load_json_dataset(train_path, val_path)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

def formatting_prompt_func_QQP(example):
    return (
        "<Instruct>Do these two questions convey the same meaning? (answer duplicate or not_duplicate)</Instruct>\n"
        "<Format>Question pair format</Format>\n"
        f"<Question>Question 1: {example['question1']}\nQuestion 2: {example['question2']}</Question>\n"
        f"<Answer>{example['label']}</Answer>"
    )

training_args = SFTConfig(
    output_dir="./sft_output_WNLI",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",               # Save model every 'n' steps instead of epochs
    #save_steps=500,                       # Save every 500 steps (adjust as needed)
    save_total_limit=1,                   # Keep only the last 2 checkpoints
    load_best_model_at_end=True,          # Load the best model at the end based on eval loss or accuracy
    metric_for_best_model="eval_loss",    # Track accuracy for best model selection
    greater_is_better= False             # Higher accuracy is better
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompt_func_WNLI,
    )

trainer.train()

metrics = trainer.evaluate()

print(metrics)