import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser

from cartography.classification.squad_utils import prepare_train_dataset_qa_dynamics, \
    Indexer, replace_ids, DynamicsTrainer, prepare_validation_dataset_qa

import os
import json

NUM_PREPROCESSING_WORKERS = 2

def main():
    argp = HfArgumentParser(TrainingArguments)
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
            This should either be a HuggingFace model ID (see https://huggingface.co/models)
            or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--dataset', type=str, default='squad',
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    training_args, args = argp.parse_args_into_dataclasses()

    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        dataset_id = tuple(args.dataset.split(':'))
        dataset = datasets.load_dataset(*dataset_id)
        eval_split = 'validation'

    model_class = AutoModelForQuestionAnswering
    model = model_class.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    prepare_train_dataset = lambda exs: prepare_train_dataset_qa_dynamics(exs, tokenizer)
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    id_indexer = Indexer()

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        train_dataset = replace_ids(train_dataset, id_indexer)
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        glue_data_dir = os.path.join(training_args.output_dir, f'glue_data/{args.dataset}')
        if not os.path.exists(glue_data_dir):
            os.makedirs(glue_data_dir)

        train_dataset.to_json(os.path.join(glue_data_dir, 'train.jsonl'))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    trainer_class = DynamicsTrainer
    eval_kwargs = {}
    metric = datasets.load_metric(args.dataset)
    eval_kwargs['eval_examples'] = eval_dataset
    compute_metrics = lambda eval_preds: metric.compute(
        predictions=eval_preds.predictions, references=eval_preds.label_ids)

    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    training_args.remove_unused_columns = False
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        id_indexer=id_indexer
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')

if __name__ == "__main__":
    main()
