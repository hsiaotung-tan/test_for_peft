import torch
from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, VeraConfig, AdaLoraConfig
from datasets import load_dataset
import numpy as np
import evaluate
import os
import pdb

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
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


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


peft_model_name = 'roberta-base-peft'
base_model = '/home/cver4090/Project/Pretrained/RoBERTa/base'
tokenizer = RobertaTokenizer.from_pretrained(base_model)

datasets = load_dataset("/home/cver4090/Project/DATA/GLUE", 'cola')
class_names = datasets["train"].features["label"].names
label_list = datasets["train"].features["label"].names
num_labels = len(label_list)

id2label = {i: label for i, label in enumerate(class_names)}


sentence1_key, sentence2_key = task_to_keys['cola']
padding = "max_length"
label_to_id = None
max_seq_length = 512

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=True, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

datasets = datasets.map(preprocess_function, batched=True)
train_dataset = datasets["train"]
eval_dataset = datasets["validation"]
test_dataset = datasets["test"]
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
# use the same Training args for all models
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-2,
    num_train_epochs=80,
    per_device_train_batch_size=64,
    save_steps=10000,
    lr_scheduler_type='linear',
    warmup_ratio=0.06,
    gradient_accumulation_steps=1,
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

# print('PEFT Model')
# peft_model.print_trainable_parameters()
# print(peft_model.peft_config)

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
# print_trainable_parameters(peft_model)

peft_trainer = get_trainer(peft_model)
# print(training_args)
# pdb.set_trace()
peft_trainer.train()
# peft_trainer.evaluate(eval_dataset=test_dataset)


