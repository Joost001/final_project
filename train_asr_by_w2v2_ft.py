

import json
import math
import os
from dataclasses import dataclass
from typing import Union, List, Dict

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, disable_progress_bar, enable_progress_bar, Audio
from evaluate import load as load_metric
from transformers import AutoConfig, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import EarlyStoppingCallback, logging, Trainer, TrainingArguments


def preprocess_text(dataset_dict):
    def __extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = sorted(list(set(all_text)))
        return {"vocab": [vocab], "all_text": [all_text]}

    def __create_vocab(dataset_dict, word_delimiter_token="|", special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
        vocab_list = []
        for ds_name, ds_data in dataset_dict.items():
            vocab = ds_data.map(__extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                remove_columns=ds_data.column_names)
            vocab_list.extend(vocab["vocab"][0])
        vocab_list = sorted(list(set(vocab_list)))
        dict_vocab = {v: k for k, v in enumerate(vocab_list)}
        dict_vocab[word_delimiter_token] = dict_vocab[" "]
        del dict_vocab[" "]
        for t in special_tokens:
            dict_vocab[t] = len(dict_vocab)
        dict_vocab = dict(sorted(dict_vocab.items(), key=lambda item: item[1]))
        return dict_vocab

    print("Pre-processing transcriptions ...")
    vocab_path = __create_vocab(dataset_dict)
    return dataset_dict, vocab_path


def dataset_from_dict(dataset_dict):
    data_set = DatasetDict()
    for k in dataset_dict.keys():
        data_set[k] = Dataset.from_pandas(pd.read_csv(dataset_dict[k], sep='\t'))
    return data_set


def configure_w2v2_for_training(pretrained_model, output_dir, use_target_vocab, dict_vocab, config_w2v2=None):
    if config_w2v2 is None:
        config_w2v2 = {}

    feature_extractor_kwargs = config_w2v2["feature_extractor"] if "feature_extractor" in config_w2v2.keys() else {}
    model_kwargs = config_w2v2["model_kwargs"] if "model_kwargs" in config_w2v2.keys() else {}
    if use_target_vocab is True:
        vocab_path = os.path.join(output_dir, 'vocab.json')
        print(f"Writing created vocabulary to {vocab_path}")
        with open(vocab_path, 'w') as vocab_file:
            json.dump(dict_vocab, vocab_file)
        AutoConfig.from_pretrained(pretrained_model).save_pretrained(output_dir)
        tokenizer = Wav2Vec2CTCTokenizer(vocab_path)
    else:
        print("Using vocabulary from tokenizer ...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model)
    feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_kwargs)
    w2v2_processor = Wav2Vec2Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

    w2v2_processor.save_pretrained(output_dir)

    pad_token_id = w2v2_processor.tokenizer.pad_token_id

    if use_target_vocab:
        model_ = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                pad_token_id=pad_token_id,
                                                vocab_size=len(w2v2_processor.tokenizer),
						ignore_mismatched_sizes=True,
                                                **model_kwargs)
    else:
        model_ = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path=pretrained_model, **model_kwargs)
    model_.freeze_feature_encoder()
    return model_, w2v2_processor


def process_data(dataset_dict, processor2):
    print("Processing data ...")

    def __helper(batch, processor=processor2):
        audio = batch["path"]
        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        # 2022-03-09:
        # Comment out input_length, not sure what actually requires this column
        # But including it results in a warning from Wav2Vec2ForCTC.forward
        # batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset_dict = dataset_dict.cast_column("path", Audio(sampling_rate=16_000))
    for ds_name, ds_data in dataset_dict.items():
        dataset_dict[ds_name] = ds_data.map(__helper, remove_columns=ds_data.column_names)
    return dataset_dict


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt", )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def get_metrics_computer(processor):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        if type(processor).__name__ == "Wav2Vec2ProcessorWithLM":
            pred_str = processor.batch_decode(pred_logits).text
        else:
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred_str = processor.batch_decode(pred_ids)

        # Replace data collator padding with tokenizer's padding
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        # Retrieve labels as characters, e.g. 'hello', from label_ids, e.g. [5, 3, 10, 10, 2] (where 5 = 'h')
        label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        print(pd.DataFrame({"pred_str": pred_str, "label_str": label_str}))

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
    return compute_metrics


def main(pretrained_model, output_dir, train_tsv, eval_tsv, use_target_vocab, hft_logging):
    logging.set_verbosity(hft_logging)

    os.makedirs(output_dir, exist_ok=True)

    dataset = dataset_from_dict({'train': train_tsv, 'eval': eval_tsv})
    w2v2_config = {"feature_extractor": {"return_attention_mask": True},
                   "model_kwargs": {"mask_time_prob": 0, "gradient_checkpointing": True, "ctc_loss_reduction": "mean"}}

    dataset, vocab_dict = preprocess_text(dataset)
    model, processor = configure_w2v2_for_training(pretrained_model, output_dir, use_target_vocab, vocab_dict, w2v2_config)

    # lm_arpa is not None:
    #     processor = configure_lm(processor, args.lm_arpa, output_dir)

    dataset = process_data(dataset, processor)

    # Set logging to 'INFO' or else progress bar gets hidden
    logging.set_verbosity(20)

    n_epochs = 20
    batch_size = 4

    # How many epochs between evals?
    eps_b_eval = 5
    # Save/Eval/Logging steps
    sel_steps = int(math.ceil(len(dataset['train']) / batch_size) * eps_b_eval)

    # commented out for testing
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=n_epochs,
        fp16=True if torch.cuda.is_available() else False,
        seed=7135,
        save_steps=sel_steps,
        eval_steps=sel_steps,
        logging_steps=sel_steps,
        learning_rate=1e-4,
        # Warm up: 100 steps or 10% of total optimisation steps
        warmup_steps=min(100, int(0.1 * sel_steps * n_epochs)),
        report_to="none",
        # 2022-03-09: manually set optmizier to PyTorch implementation torch.optim.AdamW
        # 'adamw_torch' to get rid of deprecation warning for default optimizer 'adamw_hf'
        optim="adamw_torch",
        metric_for_best_model="wer",
        save_total_limit=5,
        load_best_model_at_end=True,
        # Lower WER is better
        greater_is_better=False)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        args=training_args,
        compute_metrics=get_metrics_computer(processor=processor),
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    print("Training model ...")
    trainer.train()


if __name__ == '__main__':
    main(pretrained_model="facebook/wav2vec2-large-robust-ft-swbd-300h",
         output_dir='./data/asr-temp',
         train_tsv='./data/wav_split_gold/train.tsv',
         eval_tsv='./data/wav_split_gold/eval.tsv',
         use_target_vocab=True,
         hft_logging=40)
