#!/usr/bin/env python3

from dataclasses import dataclass
import json
import random
import time

random.seed(42)  # noqa: E402

from datasets import DatasetDict, load_from_disk
from evaluate import load
from loguru import logger
import numpy as np
import polars as pl
from rich.pretty import pprint
from safetensors.torch import save_file as safe_save_file
import torch
from transformers import (
	Trainer,
	TrainingArguments,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2FeatureExtractor,
	Wav2Vec2ForCTC,
	Wav2Vec2Processor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import typer

from stt_amh.config import CV_DATASET, MODELS_DIR

app = typer.Typer()


def load_dataset() -> DatasetDict:
	logger.info("Loading the dataset")

	return load_from_disk(CV_DATASET)


def show_random_elements(dataset: list[str], num_examples=10):
	assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
	picks = []
	for _ in range(num_examples):
		pick = random.randint(0, len(dataset) - 1)
		while pick in picks:
			pick = random.randint(0, len(dataset) - 1)
		picks.append(pick)

	df = pl.DataFrame([dataset[i] for i in picks])
	with pl.Config(
		tbl_cols=-1,
		fmt_str_lengths=500,
	):
		print(df)


# Build the full vocab set
def extract_all_chars(batch):
	all_text = " ".join(batch["sentence"])
	vocab = list(set(all_text))
	return {"vocab": [vocab], "all_text": [all_text]}


@dataclass
class DataCollatorCTCWithPadding:
	"""
	Data collator that will dynamically pad the inputs received.
	Args:
		processor (:class:`~transformers.Wav2Vec2Processor`)
		The processor used for proccessing the data.
		padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
		Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
		among:
		* :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
			sequence if provided).
		* :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
			maximum acceptable input length for the model if that argument is not provided.
		* :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
			different lengths).
	"""

	processor: Wav2Vec2Processor
	padding: bool | str = True

	def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
		# split inputs and labels since they have to be of different lengths and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)

		labels_batch = self.processor.pad(
			labels=label_features,
			padding=self.padding,
			return_tensors="pt",
		)

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch


class TrainContext:
	"""Convenient container for everything we need. Mainly to allow us to cleanly separate the steps in small methods with sane signatures."""

	def __init__(self) -> None:
		self.target_lang = "amh"
		self.run_dir = MODELS_DIR / f"amh-{time.strftime('%Y%m%d_%H%M%S')}"
		self.run_dir.mkdir(parents=True, exist_ok=True)

	def setup(self) -> None:
		self.dataset = load_dataset()
		self.build_vocab()
		self.print_vocab()
		self.dump_vocab()

	def build_vocab(self) -> None:
		logger.info("Extracting the vocabulary")

		self.vocab_train = self.dataset["train"].map(
			extract_all_chars,
			batched=True,
			batch_size=-1,
			keep_in_memory=True,
			remove_columns=self.dataset["train"].column_names,
		)
		self.vocab_test = self.dataset["test"].map(
			extract_all_chars,
			batched=True,
			batch_size=-1,
			keep_in_memory=True,
			remove_columns=self.dataset["test"].column_names,
		)
		self.vocab_validation = self.dataset["validation"].map(
			extract_all_chars,
			batched=True,
			batch_size=-1,
			keep_in_memory=True,
			remove_columns=self.dataset["validation"].column_names,
		)

	def print_vocab(self) -> None:
		vocab_list = list(
			set(self.vocab_train["vocab"][0]) | set(self.vocab_test["vocab"][0]) | set(self.vocab_validation["vocab"][0])
		)
		vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
		vocab_dict["|"] = vocab_dict[" "]
		del vocab_dict[" "]

		vocab_dict["[UNK]"] = len(vocab_dict)
		vocab_dict["[PAD]"] = len(vocab_dict)

		pprint(vocab_dict)

		self.vocab_dict = vocab_dict

	def dump_vocab(self) -> None:
		new_vocab_dict = {self.target_lang: self.vocab_dict}
		with (self.run_dir / "vocab.json").open("w") as f:
			json.dump(new_vocab_dict, f)

	def load_pipeline(self) -> None:
		self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
			self.run_dir,
			unk_token="[UNK]",
			pad_token="[PAD]",
			word_delimiter_token="|",
			target_lang=self.target_lang,
			device_map="auto",
		)

		"""
		A Wav2Vec2FeatureExtractor object requires the following parameters to be instantiated:

		- `feature_size`: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal 22.
		- `sampling_rate`: The sampling rate at which the model is trained on.
		- `padding_value`: For batched inference, shorter inputs need to be padded with a specific value
		- `do_normalize`: Whether the input should be zero-mean-unit-variance normalized or not. Usually, speech models perform better when normalizing the input
		- `return_attention_mask`: Whether the model should make use of an attention_mask for batched inference. In general, XLS-R models checkpoints should always use the attention_mask.
		"""
		self.feature_extractor = Wav2Vec2FeatureExtractor(
			feature_size=1,
			sampling_rate=16000,
			padding_value=0.0,
			do_normalize=True,
			return_attention_mask=True,
			device_map="auto",
		)

		self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

	def prepare_dataset(self) -> None:
		def prepare_dataset(batch):
			audio = batch["audio"]

			# batched output is "un-batched"
			batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
			batch["input_length"] = len(batch["input_values"])

			batch["labels"] = self.processor(text=batch["sentence"]).input_ids

			return batch

		logger.info("Preparing the dataset...")
		self.dataset["train"] = self.dataset["train"].map(
			prepare_dataset, remove_columns=self.dataset["train"].column_names
		)
		self.dataset["test"] = self.dataset["test"].map(prepare_dataset, remove_columns=self.dataset["test"].column_names)
		self.dataset["validation"] = self.dataset["validation"].map(
			prepare_dataset, remove_columns=self.dataset["validation"].column_names
		)

	def train(self) -> None:
		logger.info("Setting up trainer...")
		data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
		wer_metric = load("wer")

		def compute_metrics(pred):
			pred_logits = pred.predictions
			pred_ids = np.argmax(pred_logits, axis=-1)

			pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

			pred_str = self.processor.batch_decode(pred_ids)
			# we do not want to group tokens when computing the metrics
			label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

			wer = wer_metric.compute(predictions=pred_str, references=label_str)

			return {"wer": wer}

		# NOTE: Hyperparameters have been left mostly as-is
		self.model = Wav2Vec2ForCTC.from_pretrained(
			"facebook/mms-1b-all",
			attention_dropout=0.0,
			hidden_dropout=0.0,
			feat_proj_dropout=0.0,
			layerdrop=0.0,
			ctc_loss_reduction="mean",
			pad_token_id=self.processor.tokenizer.pad_token_id,
			vocab_size=len(self.processor.tokenizer),
			ignore_mismatched_sizes=True,
			device_map="auto",
		)

		self.model.init_adapter_layers()

		# Freeze all weights, but the adapter layers
		self.model.freeze_base_model()

		adapter_weights = self.model._get_adapters()
		for param in adapter_weights.values():
			param.requires_grad = True

		self.trainer_output_dir = self.run_dir / "wav2vec2-large-mms-1b-amharic-cv"
		training_args = TrainingArguments(
			output_dir=self.trainer_output_dir,
			train_sampling_strategy="group_by_length",
			per_device_train_batch_size=24,  # Lowered from 32 to lower memory pressure
			gradient_accumulation_steps=1,  # NOTE: This could be used as another lever to reduce memory pressure some more
			eval_strategy="steps",
			num_train_epochs=24,
			gradient_checkpointing=True,
			fp16=True,
			save_steps=200,
			eval_steps=100,
			logging_steps=100,
			learning_rate=1e-3,
			warmup_steps=100,
			save_total_limit=2,
			push_to_hub=False,
		)

		self.trainer = Trainer(
			model=self.model,
			data_collator=data_collator,
			args=training_args,
			compute_metrics=compute_metrics,
			train_dataset=self.dataset["train"],
			eval_dataset=self.dataset["validation"],
			processing_class=self.processor.feature_extractor,
		)

		logger.info("Starting training...")
		train_output = self.trainer.train()
		logger.info("Done training!")
		pprint(train_output)

	def save_adapter(self) -> None:
		logger.info("Saving trained adapter to disk...")
		save_dir = self.run_dir / "mms-1b-amh"
		save_dir.mkdir(parents=True, exist_ok=True)

		adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(self.target_lang)
		adapter_file = save_dir / adapter_file

		safe_save_file(self.model._get_adapters(), adapter_file, metadata={"format": "pt"})

		# Actually save the whole thing, as i'm not sure mixing our adapter with the stock model quite works...
		logger.info("Saving model to disk...")
		self.trainer.save_model(save_dir)
		self.processor.save_pretrained(save_dir)


@app.command()
def main() -> None:
	ctx = TrainContext()
	# Load the data & build the vocab for the tokenizer
	ctx.setup()

	logger.info("Sanity-checking a few text samples:")
	show_random_elements(ctx.dataset["train"]["sentence"], num_examples=10)

	ctx.load_pipeline()

	logger.info("Sanity-checking a few audio samples:")
	print(ctx.dataset["train"][0]["audio"])
	rand_int = random.randint(0, len(ctx.dataset["train"]) - 1)

	print("Target text:", ctx.dataset["train"][rand_int]["sentence"])
	print("Input array shape:", ctx.dataset["train"][rand_int]["audio"]["array"].shape)
	print("Sampling rate:", ctx.dataset["train"][rand_int]["audio"]["sampling_rate"])

	# Wrangle the data as expected by the trainer
	ctx.prepare_dataset()

	# Do the thing!
	ctx.train()

	# Save the thing ;)
	ctx.save_adapter()


if __name__ == "__main__":
	app()
