# Dated Data: Tracing Knowledge Cutoffs in Large Language Models

This repository accompanies the paper [Dated Data: Tracing Knowledge Cutoffs in Large Language Models](https://arxiv.org/abs/2403.12958) accepted to COLM 2024.

## Overview

Large Language Models (LLMs) are often paired with a reported cutoff date, the time at which training data was gathered. However, does the model's demonstrated knowledge align to its cutoff date? We define the notion of an effective cutoff, which indicates when the model's knowledge is most concentrated, and is different from the reported cutoff. We propose a simple approach to estimate effective cutoffs of an LLM on the resource-level by probing across versions of the data. Crucially, our method does not require access to a model's pre-training data. Through our analysis, we find that effective cutoffs often drastically differ from reported cutoffs. This repository contains our results, as well as the code to replicate them.

![alt text](./images/timestamp.drawio.pdf)

## Data Collection

We provide a .csv file of the 5000 most popular Wikipedia topics used in our analysis. Additionally, the pipeline to scrape the versions of those topics can be run as follows:

```
  python get_revision_ids.py {{csv file of most popular topics}}
  python get_content_for_month.py {{csv file of revision ids}} {{location to save scraped Wikipedia documents}}
```

## Perplexity Measurements

We provide the perplexities of the versions of the 5000 most popular Wikipedia topics spanning from 2016 - 2023 in `./perplexities/` separated for each model. Moreover, we provide the code used to generate these perplexities. Note that the data in the provided .csv files are the averaged negative log likelihoods, exponentiate to get perplexity. Fill out `config.yaml` with the relevant paths and run:

```
  python get_ppls.py --config_file ./config.yaml
```
