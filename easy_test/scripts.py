import torch
from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, VeraConfig, AdaLoraConfig
from datasets import load_dataset
import numpy as np
import evaluate
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import TrainerCallback
metric_str = "matthews_correlation"
metric = evaluate.load(metric_str)

class BestMetricCallback(TrainerCallback):
    """A custom callback that reports the best evaluation metrics across epochs."""
    def __init__(self):
        super().__init__()
        self.best_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Loop through each metric in the current evaluation metrics
        for key, value in metrics.items():
            # Check if we should update the best metric for each key
            if key.startswith('eval_'+metric_str):
                if key not in self.best_metrics or value > self.best_metrics[key]:  # Assuming lower is better; change condition for accuracy or other metrics
                    self.best_metrics[key] = value
                    print(f"New best {key}: {value}")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training completed. Best metrics:")
        for key, value in self.best_metrics.items():
            print(f"{key}: {value}")

peft_model_name = 'roberta-base-peft'
base_model = 'roberta-base'
# base_model = '/home/cver4090/Project/Pretrained/RoBERTa/base'

dataset = load_dataset('/home/cver4090/Project/DATA/GLUE', 'cola')
print(dataset)
tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess(examples):
    tokenized = tokenizer(examples['sentence'], truncation=True, padding=True, max_length=512)
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True,  remove_columns=["sentence"])
print(tokenized_dataset)
train_dataset=tokenized_dataset['train']
# NOTE: why use the same data
eval_dataset=tokenized_dataset['validation'].shard(num_shards=2, index=0)
test_dataset=tokenized_dataset['validation'].shard(num_shards=2, index=1)


# Extract the number of classess and their names
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
# We will need this for our classifier.
id2label = {i: label for i, label in enumerate(class_names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# use the same Training args for all models
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=1e-2,
    num_train_epochs=80,
    per_device_train_batch_size=64,
    save_steps=10000,
    lr_scheduler_type='linear',
    warmup_ratio=0.06,
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    eval_metric = metric.compute(predictions=predictions, references=labels)
    print(eval_metric)
    return eval_metric

def get_trainer(model):
      return  Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          callbacks=[BestMetricCallback()]
      )


model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)

peft_config = VeraConfig(task_type="SEQ_CLS", inference_mode=False, r=1024, target_modules=["query", "value"],)
peft_model = get_peft_model(model, peft_config)

print('PEFT Model')
peft_model.print_trainable_parameters()
print(peft_model.peft_config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(param.shape)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params-768*771-2} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
print_trainable_parameters(peft_model)

# peft_trainer = get_trainer(peft_model)

# peft_trainer.train()
# peft_trainer.evaluate()


