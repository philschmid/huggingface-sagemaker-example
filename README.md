# huggingface-sagemaker-example

This Repository contains multiple examples how to use the transformers and datasets library from HuggingFace. All Notebooks can be run locally or with in AWS Sagemaker Studio.

**structure**

Each folder starting with `0X_..` contains an specific sagemaker example. Each example contains a jupyter notebooke `sagemaker-example.ipynb` and a `src/` folder. The `sagemaker-example` is a jupyter notebook which is used to train transformers and datasets on AWS Sagemaker. The `src/` folder contains the `train.py`, our training script and `requirements.txt` for additional dependencies.

`notebook_template/` contains the original colab notebook, which is the baseline for the sagemaker example.

# How to use Sagemaker Data Studio
