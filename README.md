# Finetune Ollama Model 🦙

<p align="center">
  <img src="https://github.com/user-attachments/assets/229d7fe1-63a3-4fd9-abb5-5b0f2348caa5" alt="ollama" width="250">
</p>

<p>You will learn how to do <b>data prep</b> and import a CSV, how to <b>train</b>, how to <b>export to Ollama!</b></p>
<p></p><b>Unsloth</b> now allows you to automatically finetune and create a <b>Modelfile.</b></p><br>

<p><b>Note:</b> This tutorial need to run on Google Collab or Jupyter Notebook.</p>

<h3><b>STEP 1 - Installation dependencies</b> 🔗</h3>
<p>To install Unsloth on your own computer, follow the installation instructions on our Github page <a href="https://github.com/unslothai/unsloth">here</a> </p>

```bash
pip install unsloth
# Also get the latest nightly Unsloth!
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

<h3><b>STEP 2 - Download model from Huggingface</b> 📥</h3>
<p>Unsloth support Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermes etc</p>

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from pprint import pprint
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    # More models at https://huggingface.co/unsloth
)
print("Model and tokenizer loaded successfully!")
```

<p>This code configures LoRA (Low-Rank Adaptation) for a language model using a PEFT (Parameter-Efficient Fine-Tuning) framework, such as Hugging Face PEFT, to enable efficient fine-tuning with reduced resource usage (e.g., VRAM, batch size).</p>

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

<h3><b>STEP 3 - Data Preparation</b> 🗂️</h3>
<p>We now use the Alpaca dataset from <a href="https://huggingface.co/datasets/vicgalle/alpaca-gpt4">vicgalle</a>, which is a version of 52K of the original <a href="https://crfm.stanford.edu/2023/03/13/alpaca.html">Alpaca dataset</a> generated from GPT4. You can replace this code section with your own data prep.</p>

```python
# Load dataset from the provided file path
dataset = load_dataset("vicgalle/alpaca-gpt4", split = "train")
# if use JSON format use this >> dataset = load_dataset("json", data_files="xxx.json", split="train")

# Check the structure of the dataset
print(dataset.column_names)
pprint(dataset[0])
```

<p><b>to_sharegpt</b>: Converts a dataset with varying structures into the ShareGPT format, which is suited for training models to handle instructions (input) and responses (output) in a conversational context.</p>
<p><b>standardize_sharegpt</b>: Standardizes datasets in the ShareGPT format to ensure consistent structure and formatting. Handles tasks like fixing incomplete data or rearranging message sequences to better suit conversational models.</p>

```python
# Convert the dataset to the format expected by ShareGPT
dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # Select more to handle longer conversations
)

# Standardize the ShareGPT dataset
dataset = standardize_sharegpt(dataset)
```
