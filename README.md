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

## Usage

### Running the Notebook
1. Open `GroMo_ASSIGNMENT.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure all dependencies are installed (refer to the notebook for the list of required packages).
3. Execute the cells in sequence to load the model, configure the Gradio interface, and launch the chatbot.

### Interacting with the Chatbot
- After running the notebook, a temporary public URL (e.g., `https://98ab2e61f19b9fa3cf.gradio.live`) will be generated.
- Access this URL in your browser to interact with the chatbot.
- Ask questions about GroMo, and the chatbot will respond using the fine-tuned model.

**Example Query**:
- What is Gromo?
**Sample Response**:
- Gromo is a financial wellness platform that provides users with personalized advice, budgeting tools, and investment opportunities.


> **Note**: The public URL expires after 72 hours. For persistent hosting, consider deploying to Hugging Face Spaces.

---

## Project Structure

The `GroMo_ASSIGNMENT.ipynb` notebook is organized as follows:

- **Environment Setup**: Imports libraries and installs dependencies.
- **Data Preprocessing**: Loads, cleans, and formats the GroMo FAQ dataset.
- **Model Fine-Tuning**: Configures and fine-tunes the LLaMA-2-7b model with QLoRA.
- **Model Inference**: Tests the model with sample queries and responses.
- **Model Evaluation**: Calculates metrics such as loss and perplexity.
- **Gradio Interface**: Sets up and launches the interactive chatbot.

---

## Dataset

The dataset, `gromo-faq-v1-0.csv`, is located in `/kaggle/input/gromo-data/` and contains 763 question-answer pairs related to GroMo. Preprocessing steps include:

- Removing HTML tags.
- Normalizing quotes and whitespace.
- Dropping invalid or duplicate entries.
- Formatting into a prompt structure: `<s>[INST] {question} [/INST] {answer} </s>`.

The dataset is divided into:
- **Training Set**: 686 samples
- **Evaluation Set**: 77 samples

---

## Model

The chatbot is based on the LLaMA-2-7b model (`meta-llama/Llama-2-7b-hf`), fine-tuned with QLoRA for efficiency. Key details:

- **Quantization**: Uses 8-bit quantization via `BitsAndBytesConfig` to optimize memory usage.
- **LoRA Configuration**: Applies efficient fine-tuning with `r=8`, `lora_alpha=16`, targeting modules `["q_proj", "k_proj", "v_proj", "o_proj"]`, and `lora_dropout=0.1`.
- **Training**: Fine-tuned over 3 epochs with a learning rate of 2e-4, using the Hugging Face `Trainer`.
- **Evaluation**: Tested on the evaluation set, achieving an evaluation loss of 1.2682 and a perplexity of 3.55.

The fine-tuned model is hosted on Hugging Face Hub: [Yashverms/gromo-llama2-faq-model](https://huggingface.co/Yashverms/gromo-llama2-faq-model).

---

## Results

The model performs effectively, with the following metrics:

- **Evaluation Loss**: 1.2682
- **Perplexity**: 3.55

**Sample Interactions**:
- **Query**: "My payout is not yet transferred"  
  **Response**: "Please ensure that you have shared your Bank account details with us. Visit 'My Profile' --> 'Banking Details'. Fill in the Bank details and verify..."
- **Query**: "What are GroMo Points?"  
  **Response**: "GroMo Points are rewards earned through various activities on the platform. 1 GroMo Point is equal to ₹1."

> **Note**: Some responses may be repetitive or generic, highlighting areas for improvement.

---

## Future Improvements

- **Enhance Response Diversity**: Reduce repetition by refining prompt engineering or expanding training data.
- **User Interface**: Improve the Gradio interface, e.g., by adding conversation history.
- **Model Experimentation**: Explore alternative architectures or fine-tuning methods for better accuracy.
- **Expand Dataset**: Incorporate additional FAQs to broaden query coverage.

---

## Contributing

Contributions are encouraged! To get involved:

1. Fork the repository.
2. Create a branch for your feature or fix.
3. Commit your changes with clear messages.
4. Push to your fork.
5. Submit a pull request to the main repository.

Report issues via [GitHub Issues](https://github.com/yasboop/Llama2-QLora-fineturned/issues).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
