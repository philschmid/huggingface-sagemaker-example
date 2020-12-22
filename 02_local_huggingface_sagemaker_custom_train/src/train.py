from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import logging
import sys
import argparse
import os
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer,AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import numpy as np

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--model-name', type=str)
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

    # creates dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # puts model on device
    model.to(device)
    
    # puts model in train mode
    model.train()

    # optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    # train loop
    for epoch in tqdm(args.epochs, desc="training"):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    # puts model into evaluation mode
    model.eval()

    # custom compute metrics for evaluation
    def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))


    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    # evaluation loop
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)


    # compute evluation metric
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        logger.info("***** Eval results  *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            
            
    # Saves the model to s3
    model.save_pretrained(args.model_dir) 

