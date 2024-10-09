
# README - Llama 3 Quantization with HQQ and Fine-Tuning

## Overview

This project implements 1-bit and 2-bit quantization for **Llama 3 models (8B and 70B)** using **HQQ (Highly Quantized Quantization)** and performs fine-tuning with **HQQ+**. The quantization reduces memory consumption and computational costs while maintaining model performance.

## Features
- **1-bit and 2-bit Quantization**: Efficient compression of large Llama 3 models.
- **Fine-Tuning with HQQ+**: Adapter-based fine-tuning to improve model accuracy after quantization.
- **Memory Efficient**: Reduced memory consumption by up to 50% for the 70B model and more for smaller models.

## Installation

To set up the environment for this project, install the required packages:

```bash
pip install hqq bitsandbytes transformers peft accelerate datasets trl
```

## Usage

### 1-bit Quantization:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

model_id = "meta-llama/Meta-Llama-3-8B"
quant_config = HqqConfig(nbits=1, group_size=64, quant_zero=False)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 2-bit Fine-Tuning:
```python
peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.05)
trainer = SFTTrainer(model=model, train_dataset=ds['train'], tokenizer=tokenizer, args=training_args)
trainer.train()
```

## Acknowledgements
This project references the work on **1-bit and 2-bit Llama 3: Quantization with HQQ and Fine-tuning with HQQ+**.

More details: [1-bit and 2-bit Llama 3 Quantization](https://kaitchup.substack.com/p/1-bit-and-2-bit-llama-3-quantization)
