from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import logging
import sys
import argparse
import os

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tokenizer', type=str)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    # parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.getLevelName('INFO'), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # load dataset
    dataset = load_dataset('imdb')

    # download model and tokenizer from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    #helper tokenizer function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    test_dataset = test_dataset.shuffle().select(range(10000)) # smaller the size for test dataset to 10k 

    # sample a to small dataset for training
    train_dataset = train_dataset.shuffle().select(range(1000)) # smaller the size for test dataset to 10k 
    test_dataset = test_dataset.shuffle().select(range(50)) # smaller the size for test dataset to 10k 


    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # set format for pytorch
    train_dataset.rename_column_("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.rename_column_("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # define training args 
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy='epoch',
        logging_dir=f'{args.output_data_dir}/logs',
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

     # Saves the model to s3
    trainer.save_model(args.model_dir) 

