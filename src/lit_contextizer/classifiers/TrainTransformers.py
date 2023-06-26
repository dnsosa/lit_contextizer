"""Collection of functions for training Transformer-based models."""

# -*- coding: utf-8 -*-

import os
import re
from collections import OrderedDict


from datasets import Dataset, DatasetDict

import evaluate

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
    Trainer, TrainingArguments


def ablate_context(input_df):
    """Hide the context mention from the classifier."""
    # Replace insider-looking prepositional phrases in the insider sentence with a mask
    # Note this will cause a small error rate for other uses of 'in' in a sentence (e.g. 'in turn, ...') which are rare
    regex_pat = re.compile(r' in [a-z ]*s( |\.)', flags=re.IGNORECASE)
    input_df["rel"] = input_df["rel"].str.replace(regex_pat, ' in [MASK]s ', regex=True)
    return input_df


def train_transformers(input_df, data_id, root_out_dir, test_frac=(1.0 / 3), do_ablate_context=True,
                       truncation=True, epochs=3, batch_size=2, learning_rate=1e-6,
                       SEED=42):
    """
    Train the transformer models.

    :param input_df: input dataframe
    :param data_id: ID for the new dataset condition
    :param root_out_dir: output directory for saved models
    :param test_frac: fraction of the dataframe to be split off for test set
    :param do_ablate_context: hide the context from the BERT models?
    :param truncation: truncate?
    :param epochs: number of training epochs
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param SEED: random seed
    """
    id2label = {0: False, 1: True}
    label2id = {False: 0, True: 1}
    model_id_mapper = {"biobert": "dmis-lab/biobert-base-cased-v1.2",
                       "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"}
    # "roberta": "roberta-base"}

    if do_ablate_context:
        input_df = ablate_context(input_df)

    if test_frac == 1:
        X_test = input_df[["rel", "con"]]
        y_test = input_df["annotation"]
        X_train = X_test  # Placeholder (if test_frac = 1 --> no new training)
        y_train = y_test  # Placeholder (if test_frac = 1 --> no new training)

    else:
        X_train, X_test, y_train, y_test = train_test_split(input_df[["rel", "con"]], input_df["annotation"],
                                                            test_size=test_frac, random_state=SEED)

    trainer_list = []
    test_dataset_list = []

    for model_id in model_id_mapper.keys():

        output_dir = f"{root_out_dir}/{model_id}/{data_id}_TRAINED"

        # CREATE DATASETS
        #################

        # Loading tokenizer here because needed in data loading and model loading
        checkpoint = model_id_mapper[model_id]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Create and process the dataset objects
        train_dataset_dict = OrderedDict()
        test_dataset_dict = OrderedDict()

        df_list = []
        splits = ["Train", "Test"]
        for data_split in splits:
            if data_split == "Train":
                train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
                # Cast bools to ints
                train_df.annotation = train_df.annotation.replace(label2id)
                df_list.append(Dataset.from_pandas(train_df))
            else:
                test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
                test_df.annotation = test_df.annotation.replace(label2id)
                df_list.append(Dataset.from_pandas(test_df))

        raw_dataset_dict = dict(zip([split.lower() for split in splits], df_list))
        dataset = DatasetDict(raw_dataset_dict)

        # Correct the column names to what's expected
        dataset = dataset.rename_column("rel", "sentence1")
        dataset = dataset.rename_column("con", "sentence2")
        dataset = dataset.rename_column("annotation", "labels")

        old_column_names = dataset['train'].column_names
        old_column_names.remove('labels')

        # Tokenize
        def tokenize_data(example, tokenizer=tokenizer):
            return tokenizer(example["sentence1"], example["sentence2"], truncation=truncation)

        dataset = dataset.map(tokenize_data, batched=True, remove_columns=old_column_names)

        train_dataset_dict[data_id] = dataset['train']
        test_dataset_dict[data_id] = dataset['test']

        # CREATE TRAINER
        ################

        # Set random seeds
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if os.path.exists(output_dir):
            model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            print(f"Pre-saved model already found, loading from {output_dir}")

        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label,
                                                                       label2id=label2id,
                                                                       problem_type="single_label_classification")
            print(f"No saved model found, loading from {checkpoint}...")
        # optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

        # Load the metrics to use
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)  # noqa: B023

        # Create collator for feeding in data
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if not os.path.exists(f"{output_dir}/config.json"):
            # Train model!
            print(f"Now training the model and saving at {output_dir}")
            trainer.train()
            trainer.save_model(output_dir)

        trainer_list.append(trainer)
        test_dataset_list.append(dataset['test'])

    return trainer_list, test_dataset_list, list(model_id_mapper.keys())
