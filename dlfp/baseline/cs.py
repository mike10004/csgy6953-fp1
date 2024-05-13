import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple

import nltk.corpus
from collections import Counter
import string
from tqdm import tqdm

import dlfp.utils
import dlfp.common
from dlfp.datasets import DatasetResolver
from dlfp.running import Attempt
from dlfp.utils import PhrasePairDataset
from dlfp.results import measure_accuracy


SuggestionDict = Dict[str, List[Tuple[float, str]]]


class Words_Offline:

	def __init__(self, dataset: PhrasePairDataset):
		self.dataset = dataset
		self.hide_progress = False

	# noinspection PyMethodMayBeStatic
	def normalize(self, raw: str) -> str:
		return dlfp.utils.normalize_answer(raw)

	def all_solution(self, clues: Dict[str, int], threshold: float = 0.65) -> SuggestionDict:
		stop = nltk.corpus.stopwords.words('english') + [""]

		clue_mapping = dict()
		all_lengths = []
		for clue in clues:
			clue_mapping[clue] = list()
			if clues[clue] not in all_lengths:
				all_lengths.append(clues[clue])

		clue_statements = list(clues.keys())
		clue_vecs = dict()
		for clue in clue_statements:
			clue_vecs[clue] = [word for word in [word.strip(string.punctuation) for word in clue.lower().split()] if word not in stop]

		# print(">>> STARTING ALL CLUES FETCH (V.1).....")
		for guess_statement, raw_answer in tqdm(self.dataset, file=sys.stdout, disable=self.hide_progress):
			correct_answer = self.normalize(raw_answer)
			if len(correct_answer) not in all_lengths:
				continue

			guess_vec = Counter([word for word in [word.strip(string.punctuation) for word in guess_statement.lower().split()] if word not in stop])

			for clue in clue_statements:
				if len(correct_answer) == clues[clue]:
					clue_vec = Counter(clue_vecs[clue])

					# https://stackoverflow.com/questions/15173225/calculate-cosine-similarity-given-2-sentence-strings
					intersection = set(guess_vec.keys()) & set(clue_vec.keys())
					numerator = sum([guess_vec[x] * clue_vec[x] for x in intersection])

					sum1 = sum([guess_vec[x]**2 for x in guess_vec.keys()])
					sum2 = sum([clue_vec[x]**2 for x in clue_vec.keys()])
					denominator = math.sqrt(sum1) * math.sqrt(sum2)

					if not denominator:
						sim =  0.0
					else:
						sim = float(numerator) / denominator

					if sim > threshold:
						clue_mapping[clue].append((sim, correct_answer))

		deduplicated = {}
		for clue, suggestions in clue_mapping.items():
			suggestion_dict = defaultdict(list)
			for sim, candidate in suggestions:
				suggestion_dict[candidate].append(sim)
			suggestions = [(max(sims), candidate) for candidate, sims in suggestion_dict.items()]
			suggestions.sort(key=lambda x: x[0], reverse=True)
			deduplicated[clue] = suggestions
		clue_mapping = deduplicated
		return clue_mapping


def evaluate_valid():
	from argparse import ArgumentParser
	resolver = DatasetResolver()
	parser = ArgumentParser()
	parser.add_argument("-d", "--dataset", required=True)
	parser.add_argument("--valid-split", default="valid")
	args = parser.parse_args()
	dataset, valid_split = args.dataset, args.valid_split
	train_dataset = resolver.by_name(dataset, split="train").normalize_answers()
	valid_dataset = resolver.by_name(dataset, split=valid_split).normalize_answers()
	valid_clues = {}
	for clue, answer in valid_dataset:
		valid_clues[clue] = len(answer)
	print(len(valid_dataset), "clues total")
	print(len(valid_clues), "unique clues")
	solver = Words_Offline(train_dataset)
	solution = solver.all_solution(valid_clues)
	timestamp = dlfp.common.timestamp()
	raw_output_file = dlfp.common.get_repo_root() / "evaluations" / f"cs-solution-{dataset}-{valid_split}-{timestamp}.json"
	raw_output_file.parent.mkdir(exist_ok=True, parents=True)
	with open(raw_output_file, 'w') as ofile:
		json.dump(solution, ofile, indent=2)
	print("raw solution written to", raw_output_file)
	attempt_file = dlfp.common.get_repo_root() / "evaluations" / f"cs-attempts-{dataset}-{valid_split}-{timestamp}.csv"
	attempt_file.parent.mkdir(exist_ok=True, parents=True)
	k = 10
	with open(attempt_file, "w", newline="", encoding="utf-8") as ofile:
		csv_writer = csv.writer(ofile)
		csv_writer.writerow(Attempt.headers(k))
		for index, (clue, correct_answer) in enumerate(valid_dataset):
			suggestion_list = solution[clue]
			suggested_answers = [answer for _, answer in suggestion_list]
			try:
				rank = suggested_answers.index(correct_answer) + 1
			except ValueError:
				rank = float("nan")
			attempt = Attempt(index, clue, correct_answer, rank, len(suggested_answers), top=tuple(suggested_answers[:k]))
			csv_writer.writerow(attempt.to_row())
	print("attempts written to", attempt_file)
	accuracy_result = measure_accuracy(attempt_file)
	accuracy_table = accuracy_result.to_table()
	accuracy_table.write()


if __name__ == '__main__':
	evaluate_valid()
