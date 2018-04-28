import os

HOME_PATH = "/home/uday"

os.environ["CORE_NLP"] = HOME_PATH+"/NLP/algebra-word-probs/algebra-word-probs/stanford-corenlp-full-2018-02-27"
# os.system("export CORE_NLP=${HOME}/NLP/algebra-word-probs/algebra-word-probs/stanford-corenlp-full-2018-02-27")

os.system("python3 -m pynlp")