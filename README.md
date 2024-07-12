# Dated Data: Tracing Knowledge Cutoffs in Large Language Models

This repository accompanies the paper [Dated Data: Tracing Knowledge Cutoffs in Large Language Models](https://arxiv.org/abs/2403.12958) accepted to COLM 2024.

## Overview

Large Language Models (LLMs) are often paired with a reported cutoff date, the time at which training data was gathered. Such information is crucial for applications where the LLM must provide up-to-date information. However, a reported cutoff only scratches the surface. Do all sub-resources in the training data share the same cutoff? Does the model's demonstrated knowledge for these sub-resources closely align to their cutoff? We define the notion of an effective cutoff, which is distinct from the LLM's reported cutoff and differs between sub-resources. We propose a simple approach to estimate effective cutoffs of an LLM on the resource-level by probing across versions of the data. Crucially, our method does not require access to a model's pre-training data. Through our analysis, we find that effective cutoffs often drastically differ from reported cutoffs.To understand the root cause of this observation, we conduct a large-scale analysis on open pre-training datasets. Our analysis reveals two reasons for these inconsistencies: (1) temporal misalignments of CommonCrawl data due to non-trivial amounts of old data in new dumps; and (2) complications in LLM deduplication schemes involving semantic duplicates and lexical near-duplicates. Overall, our results show that cutoffs are not as simple as they have seemed and that care must be taken both by LLM dataset curators as well as practitioners who seek to use these models.

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
