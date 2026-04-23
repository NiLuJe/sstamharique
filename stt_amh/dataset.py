#!/usr/bin/env python3

import contextlib
from functools import partial
import re
from typing import Any

from datasets import Audio, DatasetDict, load_dataset
from loguru import logger
from rich.pretty import pprint
import typer

from stt_amh.config import CV_DATASET, CV_DATASET_DESC, CV_DATASET_DIR

app = typer.Typer()

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)


def import_dataset() -> DatasetDict:
	logger.info("Importing the raw dataset into a HF Dataset...")

	# There aren't any actual splits yet, so everything's in the default train
	dataset_full = load_dataset("csv", data_files=CV_DATASET_DESC.as_posix(), split="train")

	dataset_full = dataset_full.remove_columns(
		[
			"client_id",
			"sentence_id",
			"sentence_domain",
			"up_votes",
			"down_votes",
			"age",
			"gender",
			"accents",
			"variant",
			"locale",
			"segment",
		]
	)
	# NOTE: There is currently a bug similar to https://github.com/huggingface/datasets/pull/7800
	#       where the cast from pa.large_string fails, so cast it to pa.string first...
	dataset_full = dataset_full.cast_column("audio", Value(dtype="string").cast_column("audio", Audio(sampling_rate=16000))

	dataset = dataset_full.train_test_split(test_size=0.2, seed=42)
	dataset_val = dataset["train"].train_test_split(test_size=0.2, seed=42)
	dataset["train"] = dataset_val["train"]
	dataset["validation"] = dataset_val["test"]

	return dataset


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
	logger.info("Cleaning up the dataset...")

	PUNCT = re.compile(r"[!-:?፡።፣፤፥፦፧፨፠‘’“”‹›]")

	def remove_punctuation(example: dict[str, Any]) -> dict[str, Any]:
		# We shouldn't need to lowercase, as the alphasyllabary has no concept of caps
		example["sentence"] = PUNCT.sub("", example["sentence"])
		return example

	dataset["train"] = dataset["train"].map(remove_punctuation)
	dataset["test"] = dataset["test"].map(remove_punctuation)
	dataset["validation"] = dataset["validation"].map(remove_punctuation)

	return dataset


@app.command()
def main():
	ds = import_dataset()
	ds = preprocess_dataset(ds)

	pprint(ds)

	logger.info("Saving the final dataset to disk")
	# The paths in the CSV are relative...
	with contextlib.chdir(CV_DATASET_DIR):
		ds.save_to_disk(CV_DATASET.as_posix())


if __name__ == "__main__":
	app()
