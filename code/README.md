
# ARITHMETIC WORD PROBLEMS

## Files:

### Code:
- word_prob_solver.py
	- Main file which takes problem text as an input and solves it to fetch the required answer
- preprocess.py
	- Support file for  `word_prob_solver.py` which preprocesses the problem text and outputs simpler fragments of sentences.
- interface.py
	- A command line interface for demonstarting  `word_prob_solver.py` 
- helper.py
	- Support file for  `word_prob_solver.py` and `interface.py`
- pynlp_server.py
	- Starts the pynlp server session in a shell which is needed for parsing used in `word_prob_solver.py`

### Code/Data:
- questions_refined.txt
	- The dataset of questions used for evaluation
- answers_refined.txt
	- The answers to the questions in `questions_refined.txt` used for evaluation
- verbs3.txt
	- List of verbs with their labelled categories

### Code/Evaluation:
- eval.py
	- Script to evaluate our model
- illi_eval.py
	-  Script to evaluate Roy Roth Illinios web interface
- wolfram_eval.py
	- Script to evaluate Wolfram Alpha API

### Code/Verb_Categorization:
- verbs_set.py
	- Script to generate labelled verbs corpus

### Report:
- report.pdf
	- A detailed report in Springer LNCS format

## Execution Steps:
- Set the correct path to stanford parser jars in `pynlp_server.py` and execute it using  `python pynlp_server.py` in a seperate shell
- Set the problem text in `word_prob_solver.py` and execute it using `python word_prob_solver.py`
- Or start the command line interface using `python interface.py`

## Dependencies:
- Stanford Core NLP jars
- pynlp
- spacy
- neuralcoref
- nltk
- nltk[wordnet]
- requests
- selenium