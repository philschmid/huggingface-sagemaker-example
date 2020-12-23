# 🤗 huggingface-sagemaker-example

This Repository contains multiple examples "how to use the transformers and datasets library from HuggingFace with AWS Sagemaker". All Notebooks can be run locally or within AWS Sagemaker Studio.

**example strucute**

Each folder starting with `0X_...` contains an sagemaker example.  
Each example contains a jupyter notebooke `sagemaker-example.ipynb` and a `src/` folder. The `sagemaker-example.ipynb` is a jupyter notebook which is used to start train job on AWS Sagemaker or preprocess data. The `src/` directory contains the `train.py`, including the training script and `requirements.txt` for additional dependencies. Currently all examples are created using the `Pytorch` Estimator. The sagemaker examples also include `local-mode` examples so you can test the aws sagemaker training job before on your local machine.

`notebook_template/` contains the original colab notebook, which is the baseline for the sagemaker example.

## 🌁 Example overview

| Example                                                                                                                                                                        | Description                                                                                                                                                                                                                                                                    | notebook-link                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [01_huggingface_sagemaker_trainer_example](https://github.com/philschmid/huggingface-sagemaker-example/tree/main/01_huggingface_sagemaker_trainer_example)                     | this example uses the `Trainer` class to fine-tune a dataset. The dataset was previously processed with the `datasets` library. The preprocessing is in this example part of the training-job, so its all done in `train.py`                                                   | [here](https://github.com/philschmid/huggingface-sagemaker-example/blob/main/01_huggingface_sagemaker_trainer_example/sagemaker-notebook.ipynb)           |
| [02_huggingface_sagemaker_custom_train_loop](https://github.com/philschmid/huggingface-sagemaker-example/tree/main/02_huggingface_sagemaker_custom_train_loop)                 | this example implements a custom `train` loop to fine-tune a dataset. The dataset was previously processed with the `datasets` library. The preprocessing is in this example part of the training-job, so its all done in `train.py`                                           | [here](https://github.com/philschmid/huggingface-sagemaker-example/blob/main/02_huggingface_sagemaker_custom_train_loop/sagemaker-notebook.ipynb)         |
| [03_huggingface_sagemaker_trainer_with_data_from_s3](https://github.com/philschmid/huggingface-sagemaker-example/tree/main/03_huggingface_sagemaker_trainer_with_data_from_s3) | this example uses the `Trainer` class to fine-tune a dataset. The dataset was processed within the jupyter notebook using `datasets` library. The dataset is then uploaded to S3 and passed as a parameter into the training job. The `train.py` contains only the fine-tuning | [here](https://github.com/philschmid/huggingface-sagemaker-example/blob/main/03_huggingface_sagemaker_trainer_with_data_from_s3/sagemaker-notebook.ipynb) |

# 🚀 How to get started

As explained above, you are able to run these examples either on your local machine or in the AWS Sagemaker Studio.

## Getting started locally

If you want to use an example on your local machine, you need:

- an AWS Account
- configured AWS credentials on your local machine,
- an AWS Sagemaker IAM Role

If you don´t have an AWS account you can create one [here](https://portal.aws.amazon.com/billing/signup?nc2=h_ct&src=header_signup&redirect_url=https%3A%2F%2Faws.amazon.com%2Fregistration-confirmation#/start). To configure AWS credentials on your local machine you can take a look [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Lastly, to create an AWS Sagemaker IAM Role you can take a look [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html), beaware if you change the name of the role, you have to adjust it in the jupyter notebook. Now you have to install dependencies from the `requirements.txt` and you are good to go.

```bash
pip install -r requirements.txt
```

## Getting started with Sagemaker Studio

If you want to use an example in sagemaker studio. You can open your sagemaker studio and then clone the github repository. Afterwards you have to install dependencies from the `requirements.txt`.
