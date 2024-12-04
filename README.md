# fcfs-monkeys

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/tpimentelms/fcfs-monkeys/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/tpimentelms/fcfs-monkeys/tree/main)

This code is used to generate random text with First-come First-serve Monkeys.

## Dependencies

To install dependencies run:
```bash
$ conda env create -f scripts/environment.yml
```
Then activate this conda environment, by running:
```bash
$ source activate.sh
```

Now install the appropriate version of pytorch and the transformers library:
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ # conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
$ pip install transformers
```
Finally, download WordNet from nltk. For this, open python in the terminal and use commands:
```python
$ import nltk
$ nltk.download('omw-1.4')
```

## Analysing results


This repository comes equiped with two commands to create the plots in our paper, and to print the main results presented there:
```bash
$ make plot_results
$ make print_results
```

If someone wants to analyse the created lexicons manually, or their correlations, we have a jupyter notebook which shows how to load them.
First start a notebook:
```bash
$ jupyter notebook
```
Then open the files:
* `scr/h06_results/Notebook_AnalyseLexicons.ipynb`: To load the lexicons for a single language/seed at a time and analyse those results.
* `scr/h06_results/Notebook_PlotCorrelations.ipynb`: To load the compiled correlations for all languages/seeds and both plot those results and run permutation tests.
* `scr/h06_results/Notebook_AnalyseHomophony.ipynb`: To analyse the number of homophonous words per simulated language, as well as the average and maximum number of homphones each wordform has.


## Running the code

### Getting the data

Use [this Wikipedia tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer) to get the data and move it into `data/<wiki-code>/parsed-wiki40b.txt` file.


### Run full pipeline

To run the entire pipeline below for a single language and seed, run:
```bash
$ make LANGUAGE=<language_code> SEED=<seed>
```
where `language_code` can be any of: `en`, `fi`, `he`, `id`, `pt`, `tr`.
Similarly, seed is one of: `00`, `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08`, `09`.
Note that the seed must be written as `0x` (instead of just `x`) for the make command to work.

To run this pipeline for all languages and all seeds, you can use the command:
```bash
$ source scripts/run_all.sh
```

Finally, to plot results and print statistical tests (after running the above pipeline for all languages and 10 seeds) run:
```bash
$ make plot_results
$ make print_results
```

The above commands give a simplifie interface to run this entire pipeline.
To run specific parts of this pipeline see the commands below.


### Processing the data

To get both (1) the list of word types for training the graphotactic model; (2) the list of tokens used for our main experiments, run:
```bash
$ make get_wiki LANGUAGE=<language_code> SEED=<seed>
```
where `language_code` can be any of: `en`, `fi`, `he`, `id`, `pt`, `tr`.

To get the tokens re-clustered according to the polysemy thresholding strategy (from experiment 2), run:
```bash
$ make get_polysemy LANGUAGE=<language_code> SEED=<seed>
```

### Graphotactics Model Training and Sampling

To train the LSTM graphotactics model and evaluate it, run:
```bash
$ make train LANGUAGE=<language_code> SEED=<seed>
$ make eval LANGUAGE=<language_code> SEED=<seed>
```
To then sample from it (both with and without repetition), run:
```bash
$ make get_phonotactics LANGUAGE=<language_code> SEED=<seed>
```

### Assign Wordforms and Get FCFS Full Lexicons

Finally, to assign the sampled wordforms above to their respective natural meanings, run:
```bash
$ make assign_wordforms LANGUAGE=<language_code> SEED=<seed>
$ make compile_results LANGUAGE=<language_code> SEED=<seed>
```

## Repository Structure

The Makefile makes running this repository's entire pipeline easier.
We now explain what is the hole of each script in this repository individually.
The following scripts are directly callable in this repository, as well of it's structure:

* `src/h01_data/`: This folder handles the data preprocessing steps.
* `src/h01_data/filter_data.py`: This script filters the wikipedia data, only keeping sentences containing only characters from the language's script.
* `src/h01_data/process_tokens.py`: This script pre-process the sentences which will be in fact used during our analysis, splitting them into individual tokens.
* `src/h01_data/process_types.py`: This script pre-process the sentences which will be used to train our graphotactic models, extracting the most frequent word types from them.

* `src/h02_polysemy/`:
* `src/h02_polysemy/downsize_embs.py`: This script trains a PCA which will later be used to downsize the used BERT embeddings.
* `src/h02_polysemy/get_polyassign_code.py`: This script both defines the threshold used during our analysis in Experiment 2, as well as partitions all tokens in our data into new individual word types, following Algorithm 2 (but not assigning wordforms to them yet).
* `src/h02_polysemy/get_polysemy_entropy.py`: This script computes the polysemy per word type using BERT embeddings.

* `src/h03_train/`: This folder contains scripts to train our graphotactic model.
* `src/h03_train/train.py`: This script trains an LSTM which will be used as our graphotactic model.

* `src/h04_eval/`: This folder contains evaluation scripts to analyse our graphotactic model.
* `src/h05_eval/get_logprobs.py`: This script evaluates our LSTM in terms of its cross-entropy on held out forms.

* `src/h05_analysis/`: This folder contains the scripts which sample phonotactically plausible words from our graphotactic model, assign them to their respective meanings and calculate the desired correlations.
* `src/h05_analysis/sample_phonotactics.py`: This script samples phonotactically valid wordforms from our LSTM with or without repetition.
* `src/h05_analysis/assign_wordforms.py`: This script assings wordforms to their respective meanings.
* `src/h05_analysis/compile_results.py`: This script compiles results and actually computes correlations.

* `src/h06_results/`: Scripts in this folder are usefull for result visualisation.
* `src/h06_results/print_results.py`: Scripts prints frequency--length correlations.
* `src/h06_results/plot_correlations.py`: Scripts plots frequency--length and polysemy--length correlations.


## Extra Information

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/fcfs-monkeys/issues).
