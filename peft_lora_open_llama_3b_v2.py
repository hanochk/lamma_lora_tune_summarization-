# Databricks notebook source
# MAGIC %pip install transformers==4.31.0 datasets==2.13.0 peft==0.4.0 accelerate==0.21.0 bitsandbytes==0.40.2 trl==0.4.7

# COMMAND ----------

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from transformers.trainer_callback import TrainerCallback
import os
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import mlflow
from huggingface_hub.hf_api import HfFolder 


HUGGINGFACEHUB_API_TOKEN = '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'
HfFolder.save_token(HUGGINGFACEHUB_API_TOKEN)

"""
#HK: https://www.philschmid.de/fine-tune-flan-t5-peft
from datasets import load_dataset

# Load dataset from the hub
dataset = load_dataset("samsum")

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
"""

"""
#os.environ["MLFLOW_EXPERIMENT_NAME"] = "trainer-mlflow-demo"
#os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
# os.environ["MLFLOW_TRACKING_URI"]=""
# os.environ["HF_MLFLOW_LOG_ARTIFACTS"]="1"
"""
os.environ["MLFLOW_EXPERIMENT_NAME"] = "trainer-mlflow-demo"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
base_dir = '/content/drive/MyDrive/Colab Notebooks'

# COMMAND ----------

# MAGIC %sql
# MAGIC USE description_generator;

# Delete any models previously created
# del model, tokenizer, pipe

 #Delete any models previously created
# del accelerator

# Empty VRAM cache
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COMMAND ----------
import pandas as pd
from datasets import load_dataset , Dataset, concatenate_datasets 
create_ds = False
if create_ds:
  rd_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
  rd_df = pd.DataFrame(rd_ds['train'])

# COMMAND ----------

  rd_df['instruction'] = 'Create a detailed description for the following product: '+ rd_df['product']+', belonging to category: '+ rd_df['category']
  rd_df = rd_df[['instruction', 'description']]

# COMMAND ----------

  rd_df_sample = rd_df.sample(n=5000, random_state=42)


  # COMMAND ----------

  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:\n"""

  # COMMAND ----------

  rd_df_sample['prompt'] = rd_df_sample["instruction"].apply(lambda x: template.format(x))

  # COMMAND ----------

  rd_df_sample.rename(columns={'description': 'response'}, inplace=True)

  # COMMAND ----------

  rd_df_sample['response'] = rd_df_sample['response'] +  "\n### End"
  rd_df_sample = rd_df_sample[['prompt', 'response']]
  rd_df_sample.to_csv(os.path.join(base_dir, "product_name_to_description.csv"))
df = pd.read_csv(os.path.join(base_dir, "product_name_to_description.csv"))
# df = spark.sql("SELECT * FROM product_name_to_description").toPandas()
df['text'] = df["prompt"]+df["response"] # the format/convention  SFTTrainer object expecting to have: namely a complete prompt with the  response embeded seperated by ### Response:\n""" under the column : 'text'
df.drop(columns=['prompt', 'response'], inplace=True)
print(df)
print(df.shape)

# COMMAND ----------


dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05, seed=42)

# COMMAND ----------

target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
#or
target_modules = ['q_proj','v_proj']

lora_config = LoraConfig(
    r=8,#or r=16
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules = target_modules,
    task_type="CAUSAL_LM",
)

# base_dir = "<base_dir_location>"

per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = 'adamw_hf'
learning_rate = 1e-5
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "linear"

# COMMAND ----------

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=base_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs = 1.0, #3.0,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)
    

# COMMAND ----------

model_path = 'openlm-research/open_llama_3b_v2'

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto', load_in_8bit=True, # 8b only in QLoRa
)

# COMMAND ----------
print("load peft config")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

trainer = SFTTrainer(
    model,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args,
)
#Upcast layer norms to float 32 for stability
for name, module in trainer.model.named_modules():
  if "norm" in name:
    module = module.to(torch.float32)

# COMMAND ----------
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    total_mem =  round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)
    print('Total: ', total_mem, 'GB')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB') 
# Initiate the training process
with mlflow.start_run(run_name='llama_3b_epoch'):
  trainer.train()

"""
def plot_metric(metric_name):
    metric = pd.read_csv(
        f"mlruns/{run}/metrics/{metric_name}", sep=" ", names=["timestamp", "value", "steps"]
    )
    plt.plot(metric["steps"], metric["value"])  plot_metric("loss")
"""
# COMMAND ----------

# #https://github.com/NVIDIA/apex/issues/965
# for param in model.parameters():
#     # Check if parameter dtype is  Half (float16)
#     if param.dtype == torch.float16:
#         param.data = param.data.to(torch.float32)

# COMMAND ----------

# MAGIC %md
# MAGIC ### If loading from saved adapter

# COMMAND ----------

# dbutils.fs.ls(base_dir)

# COMMAND ----------

model_path = 'openlm-research/open_llama_3b_v2'

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    model_path, load_in_8bit=True, device_map='auto',
)

# COMMAND ----------

# peft_model_id = '<adapter_final_checkpoint_location>'

# COMMAND ----------
model.config.to_json_file('adapter_config.json')
peft_model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

test_strings = ["Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse",
"Create a detailed description for the following product: Hoover Lightspeed, belonging to category: Cordless Vacuum Cleaner",
"Create a detailed description for the following product: Flattronic Cinematron, belonging to category: High Definition Flatscreen TV"]

# COMMAND ----------

predictions = []
for test in test_strings:
  prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  {}

  ### Response:""".format(test)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

  generation_output = model.generate(
      input_ids=input_ids, max_new_tokens=156
  )
  predictions.append(tokenizer.decode(generation_output[0]))

# COMMAND ----------

def extract_response_text(input_string):
    start_marker = '### Response:'
    end_marker = '###'
    
    start_index = input_string.find(start_marker)
    if start_index == -1:
        return None
    
    start_index += len(start_marker)
    
    end_index = input_string.find(end_marker, start_index)
    if end_index == -1:
        return input_string[start_index:]
    
    return input_string[start_index:end_index].strip()

# COMMAND ----------

# predictions[2]

# COMMAND ----------

for i in range(3): 
  pred = predictions[i]
  text = test_strings[i]
  print(text+'\n')
  print(extract_response_text(pred))
  print('--------')

# COMMAND ----------


