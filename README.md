# Dated Data: Tracing Knowledge Cutoffs in Large Language Models

This repository accompanies the paper [Dated Data: Tracing Knowledge Cutoffs in Large Language Models](https://arxiv.org/abs/2403.12958), Outstanding Paper at COLM 2024.

## Overview

Large Language Models (LLMs) are often paired with a reported cutoff date, the time at which training data was gathered. However, does the model's demonstrated knowledge align to its cutoff date? We define the notion of an effective cutoff, which indicates when the model's knowledge is most concentrated, and is different from the reported cutoff. We propose a simple approach to estimate effective cutoffs of an LLM on the resource-level by probing across versions of the data. Crucially, our method does not require access to a model's pre-training data. Through our analysis, we find that effective cutoffs often drastically differ from reported cutoffs. This repository contains our results, as well as the code to replicate them.

![alt text](https://github.com/nexync/dated_data/blob/main/images/timestamp.drawio.png?raw=true)

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
## Citations

If you find this work helpful, please consider citing:

```
@misc{cheng2024dateddatatracingknowledge,
      title={Dated Data: Tracing Knowledge Cutoffs in Large Language Models}, 
      author={Jeffrey Cheng and Marc Marone and Orion Weller and Dawn Lawrie and Daniel Khashabi and Benjamin Van Durme},
      year={2024},
      eprint={2403.12958},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.12958}, 
}
```
