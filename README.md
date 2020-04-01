# Collection of Adversarial Example Generator Scripts for the SQuAD Datset

This repository contains code for the paper:

> Adversarial Examples for Evaluating Reading Comprehension Systems.  
> Robin Jia and Percy Liang  
> Empirical Methods in Natural Language Processing (EMNLP), 2017.  

## Dependencies

Run `pull-dependencies.sh` to pull SQuAD data, GloVe vectors, Stanford CoreNLP,
and some custom python utilities.
Other python requirements are in `requirements.txt`.

## Examples

### AddSent

The following sequence of commmands generates the raw AddSent training data described in Section 4.6 of the paper.

    mkdir out
    # Precompute nearby words in word vector space; takes roughly 1 hour
    python src/py/find_squad_nearby_words.py glove/glove.6B.100d.txt -n 100 -f data/squad/train-v1.1.json > out/nearby_n100_glove_6B_100d.json
    # Run CoreNLP on the SQuAD training data; takes roughly 1 hour, uses ~18GB memory
    python src/py/convert_questions.py corenlp -d train
    # Actually generate the raw AddSent examples; takes roughly 7 minutes, uses ~15GB memory
    python src/py/convert_questions.py dump-highConf -d train -q

The final script will generate three files with prefix `train-convHighConf` in the `out` directory,
including `train-convHighConf.json`. `train-convHighConf-mturk.tsv` is in a format that can be processed by scripts in the `mturk` directory.

### AddSentDiverse

    mkdir out
    # Precompute nearby words in word vector space; takes roughly 1 hour
    python src/py/find_squad_nearby_words.py glove/glove.6B.100d.txt -n 100 -f data/squad/train-v1.1.json > out/nearby_n100_glove_6B_100d.json
    # Run CoreNLP on the SQuAD training data; takes roughly 1 hour, uses ~18GB memory
    python src/py/convert_questions.py corenlp -d train
    # Actually generate the raw AddSent examples; takes roughly 7 minutes, uses ~15GB memory
    python src/py/convert_questions.py dump-highConf -d train -q

The following sequence of commmands generates the AddSentDiverse training data described Wang et. al 2018.

    # TODO
