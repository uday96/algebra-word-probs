from nltk.corpus import wordnet
import json

with open("../data/verbs3.txt") as f:
	content = f.readlines()

verbs = []
labels = []

cats = {
    '0': 'OBS',
    '1': 'OBS',
    '+': 'POS',
    '-': 'NEG',
    't-': 'NEG_TR',
    't+': 'POS_TR',
    '--': 'DESTROY',
    '++': 'CONSTRUCT'
}


for line in content:
	verbs.append(line.split()[0].strip())
	labels.append(line.split()[1].strip())

print(len(verbs))

data = {}
data2 = []

for verb in range(0,len(verbs)):
	print(verbs[verb],cats[labels[verb]])
	data[verbs[verb]]=cats[labels[verb]]
	for syn in wordnet.synsets(verbs[verb],pos="v"):
		for l in syn.lemmas():
			if l.name() not in data:
				data[l.name()]=cats[labels[verb]]
			else:
				data2.append([l.name(),cats[labels[verb]]])


#print(data)
print(len(data))

print(len(data2))

with open('verb_cats.json', 'w') as fp:
    json.dump(data, fp)