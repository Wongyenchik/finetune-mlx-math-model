# Fine-tuned LLM Model using MLX on Apple Silicon

This repository contains a fine-tuned LLM based on the Hugging Face gemma-2-2b model. The fine-tuning process leverages the MLX framework, which supports quantization and fine-tuning on Apple Silicon’s unified memory architecture. The LoRA (Low-Rank Adaptation) method is used to train a few layers, enabling the model to achieve high performance with minimal resource requirements. This README will guide you through understanding the project structure, setup, and how to run and test the model.

## Project Overview

The main components of this project include:

- **Data Preprocessing**: Converts the MATH-500 dataset from Hugging Face into JSONL format as required by the MLX framework, inserting it with the prepared prompt template.
- **Quantization** of the gemma-2-2b model using MLX.
- **Fine-tuning** using the LoRA method with the MATH-500 dataset from Hugging Face.
- **Data Storage**: Stores both the dataset and LoRA adapter layers in Amazon S3.
- **Model Evaluation** using the MLFlow framework and OpenAI API, with results stored for version control.
- **Frontend Integration** using Streamlit for testing and interacting with the model.

## Key Directories and Files

When you successfully run the finetune-model.ipynb file, a few new directories will appear in your repository:

- Mlruns: Stores previous ML training runs, details, and logs via MLFlow.
- Adapters: Contains the LoRA adapter layers from the fine-tuned model.
- Mlx_model: The quantized model from Hugging Face’s gemma-2-2b model.
- My-model: The final model, which includes the quantized base model combined with the LoRA adapter layers.
- Mlartifacts: Stores all artifacts related to the models in MLFlow for version control and tracking.

These directories are not uploaded to the repository due to size constraints but will be created when the fine-tuning process is run.

## Setup Instructions

### Prerequisites

This model requires the following specifications to run:

- Apple Silicon Mac (M-series chip)
- macOS version 13.5 or later
- Python 3.9 or higher

Additionally, you need to include the following environment variables in a .env file for the model to run successfully:

- HUGGING_FACE_TOKEN
- AWS_ACCESS_KEY
- AWS_SECRET_KEY
- AWS_REGION
- OPENAI_API_KEY

### Installation

1. Clone this repository
2. Install dependencies:
   `pip install -r requirements.txt`

Create the .env file: Add the required environment variables mentioned above in a .env file.

### Running the Fine-tuning Process

Open the `finetune-model.ipynb` file and execute all the cells. This will:

1. Preprocess the data by inserting it into the prompt template, converting it into JSONL format, and storing it in Amazon S3.
2. Quantize the gemma-2-2b model using MLX.
3. Train the LoRA layers using the MATH-500 dataset and store the results in Amazon S3.
4. Save the quantized and fine-tuned model.
5. Test the model’s performance using MLFlow and OpenAI.
6. Publish the model to Hugging Face.

### Testing the Fine-tuned Model

To test the model that I have already trained and pushed to Hugging Face, you can use the provided Streamlit app:

1. Run the Streamlit app:
   `streamlit run streamlit/app.py`
2. Interact with the model: Once the app is running, you can enter your inputs and get real-time responses from the fine-tuned model.

## Results

### Evaluation Using MLFlow

This project uses MLFlow to evaluate the fine-tuned model:

- OpenAI API Evaluation: The evaluation method uses OpenAI’s API to test the model’s performance, with results stored in MLFlow for better version control.
- Check Results: Evaluation results are stored in the `eval_results_table.json` file.

### Evaluation Results

The results of the OpenAI evaluation are stored in the `eval_results_table.json` file.

**Please note that some evaluations could not be completed due to rate limiting from the OpenAI API**

`Failed to score model on payload. Error: Failed to call LLM endpoint at https://api.openai.com/v1/chat/completions.Error: 429 Client Error: Too Many Requests for url: https://api.openai.com/v1/chat/completions.`

### Future Improvements

- **Retry Mechanism**: Implement a retry mechanism after a timeout to handle OpenAI API rate-limiting errors (Error 429).
- **Hyperparameter Tuning**: The current fine-tuning process was done with a single iteration. Future iterations should experiment with different hyperparameters to improve model performance.
- **Performance Enhancement**: The current model’s performance is suboptimal based on the evaluation results, with low correctness scores (1-2 out of 5). Further fine-tuning with multiple iterations or dataset augmentation may improve accuracy.
