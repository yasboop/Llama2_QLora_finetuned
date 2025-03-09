# GroMo FAQ Chatbot [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Yashverms/gromo-llama2-faq-model)

This project is a chatbot designed to answer frequently asked questions about GroMo, a financial wellness platform. It leverages a fine-tuned LLaMA-2-7b model with QLoRA for efficient natural language understanding and is deployed using Gradio for an interactive user interface.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Accurate FAQ Responses**: Delivers precise answers to common questions about GroMo.
- **Fine-Tuned LLaMA-2-7b Model**: Utilizes a state-of-the-art language model for natural language understanding.
- **Interactive Gradio Interface**: Enables easy user interaction through a web-based interface.
- **Evaluation Metrics**: Achieves a perplexity score of 3.55, demonstrating strong performance.

---

## Technologies Used

- **Python**: Core programming language for development.
- **Jupyter Notebook**: Interactive environment for development and documentation.
- **Hugging Face Transformers**: Used for model loading, fine-tuning, and inference.
- **PEFT (Parameter-Efficient Fine-Tuning)**: Implements LoRA for efficient fine-tuning.
- **Datasets**: Manages and processes the FAQ dataset.
- **Accelerate**: Optimizes training on accelerators like GPUs.
- **BitsAndBytes**: Enables 8-bit quantization to reduce memory usage.
- **Gradio**: Powers the interactive web interface.

---

## Installation

To set up and run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yasboop/Llama2-QLora-fineturned.git
   cd Llama2-QLora-fineturned
