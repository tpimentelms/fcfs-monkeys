LANGUAGE := fi
SEED := 07

MAX_SENTENCES_LARGE := 20000000
MAX_SENTENCES := 2000000
# MAX_SENTENCES_LARGE := 10000
# MAX_SENTENCES := 1700
N_SEEDS := 10

MAX_TRAIN_TYPES := 10000
BERT_BATCH_SIZE := 8

FCFS_TEMPERATURE := .5
IID_TEMPERATURE := 5

DATA_DIR_BASE := ./data
DATA_DIR_LANG := $(DATA_DIR_BASE)/$(LANGUAGE)
DATA_DIR := $(DATA_DIR_LANG)/seed_$(SEED)
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR := $(CHECKPOINT_DIR_BASE)/$(LANGUAGE)/seed_$(SEED)
RESULTS_DIR := ./results

# Data Files
TOKENIZED_FILE := $(DATA_DIR_LANG)/parsed-wiki40b.txt
SHUFFLED_LARGE_FILE := $(DATA_DIR_LANG)/shuffled-large.txt
FILTERED_DATA_FILE := $(DATA_DIR_LANG)/filtered.txt
SHUFFLED_FILE := $(DATA_DIR_LANG)/shuffled.txt
SHUFFLED_SPLIT_TEMP_FILE_PREFIX := $(DATA_DIR_LANG)/shuffled-split_temp

TYPE_DATA_FILE_FULL := $(DATA_DIR_LANG)/types.txt
TOKEN_DATA_FILE_FULL := $(DATA_DIR_LANG)/tokens.txt
TYPE_DATA_FILE_PREFIX := $(DATA_DIR_LANG)/types-split-
TOKEN_DATA_FILE_PREFIX := $(DATA_DIR_LANG)/tokens-split-
DATA_FILES_SUFFIX := .txt
TYPE_DATA_FILE := $(TYPE_DATA_FILE_PREFIX)$(SEED).txt
TOKEN_DATA_FILE := $(TOKEN_DATA_FILE_PREFIX)$(SEED).txt

PROCESSED_TYPE_FILE := $(DATA_DIR)/types_proc.pckl
PROCESSED_TOKEN_FILE := $(DATA_DIR)/tokens_proc.tsv

# Polysemy Files
EMB_DATA_DIR := $(DATA_DIR)/emb-orig/
CHECKPOINT_PCA_FILE := $(CHECKPOINT_DIR)/pca.pckl
CHECKPOINT_POLYASSIGN_CODE := $(CHECKPOINT_DIR)/polyassign_code.pckl
CHECKPOINT_ENTROPY_POLYASSIGN := $(CHECKPOINT_DIR)/entropy_polyassign.tsv
CHECKPOINT_ENTROPY_NATURAL := $(CHECKPOINT_DIR)/entropy_natural.tsv

# Graphotactic Model Files
CHECKPOINT_GRAPHOTACTIC_FILE := $(CHECKPOINT_DIR)/graphotactic-model.tch
CHECKPOINT_GRAPHOTACTIC_LOGPROBS := $(CHECKPOINT_DIR)/graphotactic-logprobs.pckl

# Monkey Wordform Samples
CHECKPOINT_FCFS_SAMPLES := $(CHECKPOINT_DIR)/fcfs_samples.pckl
CHECKPOINT_CAPLAN_SAMPLES := $(CHECKPOINT_DIR)/caplan_samples.pckl
CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES := $(CHECKPOINT_DIR)/caplan_low_temp_samples.pckl

# Codes and Compiled Results
CHECKPOINT_FREQ_CODES := $(CHECKPOINT_DIR)/codes.tsv
CHECKPOINT_COMPILED_NATURAL_POLYSEMY := $(CHECKPOINT_DIR)/compiled_natural_polysemy.tsv
CHECKPOINT_COMPILED_POLYASSIGN_POLYSEMY := $(CHECKPOINT_DIR)/compiled_polyassign_polysemy.tsv
CHECKPOINT_COMPILED_RESULTS := $(CHECKPOINT_DIR)/compiled_results.tsv



all: get_wiki get_polysemy train eval get_phonotactics assign_wordforms compile_results

print_results:
	python src/h06_results/print_results.py  --checkpoints-path $(CHECKPOINT_DIR_BASE)
# 	python src/h06_results/print_results.py  --checkpoints-path $(CHECKPOINT_DIR_BASE) --use-low-temperature

plot_results:
	mkdir -p $(RESULTS_DIR)
	python src/h06_results/plot_correlations.py  --checkpoints-path $(CHECKPOINT_DIR_BASE) --results-path $(RESULTS_DIR)
	python src/h06_results/plot_correlations.py  --checkpoints-path $(CHECKPOINT_DIR_BASE) --results-path $(RESULTS_DIR) --use-low-temperature

compile_results: $(CHECKPOINT_COMPILED_POLYASSIGN_POLYSEMY) $(CHECKPOINT_COMPILED_RESULTS)

assign_wordforms: $(CHECKPOINT_FREQ_CODES)

get_phonotactics: $(CHECKPOINT_FCFS_SAMPLES) $(CHECKPOINT_CAPLAN_SAMPLES) $(CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES)

eval: $(CHECKPOINT_GRAPHOTACTIC_LOGPROBS)

train: $(CHECKPOINT_GRAPHOTACTIC_FILE)

get_polysemy: $(EMB_DATA_DIR) $(CHECKPOINT_PCA_FILE) \
	$(CHECKPOINT_POLYASSIGN_CODE) $(CHECKPOINT_ENTROPY_POLYASSIGN)

get_wiki_seed: $(PROCESSED_TYPE_FILE) $(PROCESSED_TOKEN_FILE)

get_wiki: $(SHUFFLED_FILE) $(TOKEN_DATA_FILE_FULL) $(TYPE_DATA_FILE) $(TOKEN_DATA_FILE)

clean:
	rm $(PROCESSED_TYPE_FILE) $(PROCESSED_TOKEN_FILE)


####### Compile Monkey Results ##########

$(CHECKPOINT_COMPILED_RESULTS):
	python src/h05_analysis/compile_results.py --seed $(SEED) --language $(LANGUAGE) \
			--results-freq-codes-file $(CHECKPOINT_FREQ_CODES) \
			--results-ent-polyassign-file $(CHECKPOINT_COMPILED_POLYASSIGN_POLYSEMY) \
			--results-ent-natural-file $(CHECKPOINT_COMPILED_NATURAL_POLYSEMY) \
			--results-compiled-file $(CHECKPOINT_COMPILED_RESULTS)

$(CHECKPOINT_COMPILED_POLYASSIGN_POLYSEMY):
	python src/h05_analysis/compile_polysemy.py --seed $(SEED) \
			--fcfs-samples-file $(CHECKPOINT_FCFS_SAMPLES)  --caplan-samples-file $(CHECKPOINT_CAPLAN_SAMPLES) \
			--caplan-low-temperature-samples-file $(CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES) \
			--results-ent-polyassign-file $(CHECKPOINT_ENTROPY_POLYASSIGN) \
			--results-ent-natural-file $(CHECKPOINT_ENTROPY_NATURAL) \
			--results-compiled-polyassign-file $(CHECKPOINT_COMPILED_POLYASSIGN_POLYSEMY) \
			--results-compiled-natural-file $(CHECKPOINT_COMPILED_NATURAL_POLYSEMY)

$(CHECKPOINT_FREQ_CODES): $(CHECKPOINT_GRAPHOTACTIC_FILE) $(CHECKPOINT_FCFS_SAMPLES)
	python src/h05_analysis/assign_wordforms.py --seed $(SEED)  \
			--fcfs-samples-file $(CHECKPOINT_FCFS_SAMPLES)  --caplan-samples-file $(CHECKPOINT_CAPLAN_SAMPLES) \
			--caplan-low-temperature-samples-file $(CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES) \
			--tokens-file $(PROCESSED_TOKEN_FILE) --results-file $(CHECKPOINT_FREQ_CODES)

####### Sample wordforms #######

$(CHECKPOINT_FCFS_SAMPLES): | $(CHECKPOINT_GRAPHOTACTIC_FILE) $(PROCESSED_TOKEN_FILE) $(CHECKPOINT_TYPE_POLYSEMY)
	python src/h05_analysis/sample_phonotactics.py --seed $(SEED) \
			--checkpoint-path $(CHECKPOINT_DIR) --samples-file $(CHECKPOINT_FCFS_SAMPLES) \
			--tokens-file $(PROCESSED_TOKEN_FILE) --types-file $(PROCESSED_TYPE_FILE) \
			--polyassign-code-file $(CHECKPOINT_POLYASSIGN_CODE) --temperature $(FCFS_TEMPERATURE)

$(CHECKPOINT_CAPLAN_SAMPLES): | $(CHECKPOINT_GRAPHOTACTIC_FILE) $(PROCESSED_TOKEN_FILE) $(CHECKPOINT_TYPE_POLYSEMY)
	python src/h05_analysis/sample_phonotactics.py --seed $(SEED) --with-repetition \
			--checkpoint-path $(CHECKPOINT_DIR) --samples-file $(CHECKPOINT_CAPLAN_SAMPLES) \
			--tokens-file $(PROCESSED_TOKEN_FILE) --types-file $(PROCESSED_TYPE_FILE) \
			--polyassign-code-file $(CHECKPOINT_POLYASSIGN_CODE) --temperature $(IID_TEMPERATURE)

$(CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES): | $(CHECKPOINT_GRAPHOTACTIC_FILE) $(PROCESSED_TOKEN_FILE) $(CHECKPOINT_TYPE_POLYSEMY)
	python src/h05_analysis/sample_phonotactics.py --seed $(SEED) --with-repetition \
			--checkpoint-path $(CHECKPOINT_DIR) --samples-file $(CHECKPOINT_CAPLAN_LOW_TEMP_SAMPLES) \
			--tokens-file $(PROCESSED_TOKEN_FILE) --types-file $(PROCESSED_TYPE_FILE) \
			--polyassign-code-file $(CHECKPOINT_POLYASSIGN_CODE) --temperature $(FCFS_TEMPERATURE)


####### Train and Evaluate Graphotactic Model #######

# Eval type lstm models
$(CHECKPOINT_GRAPHOTACTIC_LOGPROBS): | $(CHECKPOINT_GRAPHOTACTIC_FILE)
	echo "Eval type model" $(CHECKPOINT_GENERATOR_LOGPROBS)
	python src/h04_eval/get_logprobs.py --seed $(SEED) \
		--data-file $(PROCESSED_TYPE_FILE) --eval-path $(CHECKPOINT_DIR)

# Train types model
$(CHECKPOINT_GRAPHOTACTIC_FILE): | $(PROCESSED_TYPE_FILE)
	echo "Train types model" $(CHECKPOINT_GRAPHOTACTIC_FILE)
	mkdir -p $(CHECKPOINT_DIR)
	python src/h03_learn/train.py --seed $(SEED) \
		--data-file $(PROCESSED_TYPE_FILE) --checkpoints-path $(CHECKPOINT_DIR)


####### Get BERT embeddings and Polysemy Estimates ##########

$(CHECKPOINT_ENTROPY_POLYASSIGN): | $(CHECKPOINT_POLYASSIGN_CODE)
	python -u src/h02_polysemy/get_polysemy_entropy.py --seed $(SEED) \
			--emb-dir $(EMB_DATA_DIR) --pca-file $(CHECKPOINT_PCA_FILE) \
			--polyassign-file $(CHECKPOINT_POLYASSIGN_CODE) \
			--polyassign-polysemy-file $(CHECKPOINT_ENTROPY_POLYASSIGN) \
			--natural-polysemy-file $(CHECKPOINT_ENTROPY_NATURAL)

# Get polysemy thresholded data
$(CHECKPOINT_POLYASSIGN_CODE): | $(CHECKPOINT_PCA_FILE)
	python -u src/h02_polysemy/get_polyassign_code.py --seed $(SEED) \
			--emb-dir $(EMB_DATA_DIR) --pca-file $(CHECKPOINT_PCA_FILE) \
			--polyassign-file $(CHECKPOINT_POLYASSIGN_CODE)

$(CHECKPOINT_PCA_FILE): | $(EMB_DATA_DIR)
	mkdir -p $(CHECKPOINT_DIR)
	python src/h02_polysemy/downsize_embs.py --seed $(SEED) --language $(LANGUAGE) \
			--emb-dir $(EMB_DATA_DIR) --pca-file $(CHECKPOINT_PCA_FILE)

$(EMB_DATA_DIR): | $(TOKEN_DATA_FILE)
	mkdir -p $(EMB_DATA_DIR)
	python wiki-bert/src/get_bert_embeddings.py --dump-size 5000 --batch-size $(BERT_BATCH_SIZE) \
		--wikipedia-tokenized-file $(TOKEN_DATA_FILE) --embeddings-raw-path $(EMB_DATA_DIR)


####### Process Seed Data ##########

# Get types to train graphotactic model
$(PROCESSED_TYPE_FILE):| $(TYPE_DATA_FILE)
	echo "Process data"
	mkdir -p $(DATA_DIR)
	python src/h01_data/process_types.py --seed $(SEED) \
		--wikipedia-tokenized-file $(TYPE_DATA_FILE) --data-file $(PROCESSED_TYPE_FILE) --max-types $(MAX_TRAIN_TYPES)

# Get tokens for main analysis
$(PROCESSED_TOKEN_FILE):| $(TOKEN_DATA_FILE)
	echo "Process data"
	mkdir -p $(DATA_DIR)
	python src/h01_data/process_tokens.py --seed $(SEED) \
		--wikipedia-tokenized-file $(TOKEN_DATA_FILE) --data-file $(PROCESSED_TOKEN_FILE)

####### Process Raw Data ##########

$(TYPE_DATA_FILE): | $(TYPE_DATA_FILE_FULL)
	split -n r/$(N_SEEDS) $(TYPE_DATA_FILE_FULL) $(TYPE_DATA_FILE_PREFIX) -da 2 --additional-suffix=$(DATA_FILES_SUFFIX)

$(TOKEN_DATA_FILE): | $(TOKEN_DATA_FILE_FULL)
	split -n r/$(N_SEEDS) $(TOKEN_DATA_FILE_FULL) $(TOKEN_DATA_FILE_PREFIX) -da 2 --additional-suffix=$(DATA_FILES_SUFFIX)

$(TOKEN_DATA_FILE_FULL): | $(SHUFFLED_FILE)
	split -n r/2 $(SHUFFLED_FILE) $(SHUFFLED_SPLIT_TEMP_FILE_PREFIX) -da 1
	mv $(SHUFFLED_SPLIT_TEMP_FILE_PREFIX)0 $(TYPE_DATA_FILE_FULL)
	mv $(SHUFFLED_SPLIT_TEMP_FILE_PREFIX)1 $(TOKEN_DATA_FILE_FULL)

# Shuffle Data
$(SHUFFLED_FILE): | $(FILTERED_DATA_FILE)
	echo "Shuffle wiki data"
	shuf $(FILTERED_DATA_FILE) -n $(MAX_SENTENCES) -o $(SHUFFLED_FILE)

$(FILTERED_DATA_FILE): | $(SHUFFLED_LARGE_FILE)
	python src/h01_data/filter_data.py --wikipedia-tokenized-file $(SHUFFLED_LARGE_FILE) --data-file $(FILTERED_DATA_FILE) --language $(LANGUAGE)

# Shuffle Data
$(SHUFFLED_LARGE_FILE): $(TOKENIZED_FILE)
	echo "Shuffle wiki data"
	shuf $(TOKENIZED_FILE) -n $(MAX_SENTENCES_LARGE) -o $(SHUFFLED_LARGE_FILE)
