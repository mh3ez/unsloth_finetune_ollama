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

<h3><b>STEP 4 - Customizable Chat Templates</b> 💬 </h3>
<p>The issue is the Alpaca format has 3 fields, whilst OpenAI style chatbots must only use 2 fields (instruction and response). That's why we used the <b>to_sharegpt</b> function to merge these columns into 1.</p>

```python
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)
```

<h3><b>STEP 5 - Train the model</b> 🦾</h3>
<p>Now let's use Huggingface TRL's <b>SFTTrainer!</b> More docs here: <a href="https://huggingface.co/docs/trl/sft_trainer">TRL SFT docs</a>. We do 60 steps to speed things up, but you can set <b>num_train_epochs=1</b> for a full run, and turn off <b>max_steps=None</b>. We also support TRL's <b>DPOTrainer</b>!</p>

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
```

```python
trainer_stats = trainer.train()
```

<p>Let's run the model! Unsloth makes inference natively 2x faster as well! You should use prompts which are similar to the ones you had finetuned on, otherwise you might get bad results!</p>

```python
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
messages = [                    # Change below!
    {"role": "user", "content": "Continue the fibonacci sequence! Your input is 1, 1, 2, 3, 5, 8,"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
```

<h3><b>STEP 6 - Saving, loading finetuned models</b> 💬 </h3>
<p>To save the final model as LoRA adapters, either use Huggingface's push_to_hub for an online save or save_pretrained for a local save.</p>
<p><b>[NOTE]</b> This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!</p>

```python
# model.save_pretrained("lora_model") # Local saving
# tokenizer.save_pretrained("lora_model")
model.push_to_hub("your_name/lora_model", token = "...") # Online saving
tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
```

<p>push model to Huggingface</p>

```python
model.push_to_hub_gguf("<hungingface_folder>", tokenizer, quantization_method = "q8_0", token = "...")
# quantization_method >> q4_k_m q8_0
```
