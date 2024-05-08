import sys
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple

import nltk.corpus
from collections import Counter
import string
import math
from tqdm import tqdm

import dlfp.utils
from dlfp.datasets import DatasetResolver
from dlfp.utils import PhrasePairDataset


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
