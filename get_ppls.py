import torch
import torch.nn as nn

import tqdm
import os 
import pandas as pd
import yaml

import collections
import argparse
from transformers import  AutoTokenizer, AutoModelForCausalLM

from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

def parse_args():
	parent_parser = argparse.ArgumentParser(add_help=False)
	
	parser = argparse.ArgumentParser(parents=[parent_parser])
	config = argparse.ArgumentParser(parents=[parent_parser])

	config.add_argument("--config_file", required=True)
	config_args, _ = config_args.parse_known_args()

	with open(config_args.config_file) as f:
		cfg = yaml.load(f, Loader = yaml.FullLoader)["default"]

	parser.add_argument("--model_path", type=str)
	parser.add_argument("--data_path", type=str)
	parser.add_argument("--work_dir", type=str)
	parser.add_argument("--save_file", type=str)
	parser.add_argument("--n_months", type=int)
	parser.add_argument("--cuda", type=bool)

	parser.set_defaults(**config)
	args, _ = parser.parse_known_args()

	return args
	

class WikiPerplexity():
	def __init__(self, model, tokenizer, args):

		#set device, model and tokenizer
		self.device = "cuda" if args.cuda else "cpu"

		model.resize_token_embeddings(tokenizer.vocab_size + 1)
		self.model = model.to(self.device)
		
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		self.tokenizer = tokenizer

		def parse_wiki_data(path_to_data):
			print("Parsing Wiki Data")
			topic_dict = collections.defaultdict(lambda: [])

			folder = os.scandir(path_to_data) #Folder of folders
			for sf in folder:
				subfolder = os.scandir(path_to_data + sf.name + "/")
				for file in subfolder:
					arr = file.name.split("-")
					topic = "_".join(arr[:-2])
					topic_dict[topic].append((sf.name, file.name))

			return topic_dict

		self.topic_dict = parse_wiki_data(args.data_path)
		self.path_to_data = args.data_path
		self.work_dir = args.save_path
		self.save_file = args.save_file
		self.n_months = args.n_months	# Total number of expected versions per topic

		self.save_freq = 100
		
	def get_document_perplexities(self, batch_size = 1, start_topic = None):
		print("Using ", self.device)

		pad_idx = self.tokenizer("[PAD]").input_ids[-1]

		res = collections.defaultdict(lambda: [])
		topics = []
		skip = True

		if not start_topic:
			skip = False
		
		for count, topic in tqdm.tqdm(enumerate(self.topic_dict.keys()), total= len(self.topic_dict.keys())):
			# Skip processing topics that don't have all versions
			if len(self.topic_dict[topic]) != self.n_months or skip:
				if topic == start_topic:
					skip = False
				continue
				
			batch_text = []
			for i, (date, name) in enumerate(self.topic_dict[topic]):
				full_path_to_file = self.path_to_data + date + "/" + name
				with open(full_path_to_file, encoding = 'utf-8') as f:
					text = f.read()
					
				batch_text.append(text)
				
				if len(batch_text) == batch_size or i == len(self.topic_dict[topic])-1:
					res[date] += self.get_perplexity(batch_text, pad_idx)
					batch_text = []

			topics.append(topic)

			if count % self.save_freq == 0:
				df = pd.DataFrame(data = list(zip(topics, *[res[topic] for topic in res.keys()])), columns=["Topic", *list(res.keys())])
				df.to_csv(self.local_path + self.save_file)
		
		print("Finished, saving to path")
		df = pd.DataFrame(data = list(zip(topics, *[res[topic] for topic in res.keys()])), columns=["Topic", *list(res.keys())])
		df.to_csv(self.local_path + self.save_file)
		
	def get_perplexity(self, batch_text, pad_idx):
		res = []
		max_length = 512

		encodings = self.tokenizer(batch_text, return_tensors = "pt", padding = True)
		input_ids = encodings.input_ids[:, :max_length].to(self.device)
		target_ids = input_ids.clone()

		with torch.no_grad():
			outputs = self.model(input_ids, labels = target_ids)
		
		probs = torch.log_softmax(outputs.logits, dim = 2)
		for i in range(len(batch_text)):
			check_padding = (input_ids[i] == pad_idx).nonzero()

			if len(check_padding) == 0:
				term = probs.shape[1]
			else:
				term = check_padding[0][0].item()

			t = torch.index_select(probs[i][:term-1], 1, input_ids[i][1:term])
			t = torch.diagonal(t, 0)
			nll = -torch.sum(t)/(term-1)
			res.append(nll.item())
			
		return res
	
if __name__ == "__main__":
	args = parse_args()

	# Can quantize model if doesn't fit on GPU
	model = AutoModelForCausalLM.from_pretrained(args.model_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	w = WikiPerplexity(model, tokenizer, args)
	w.get_document_perplexities(start_topic=None, batch_size = 4)
