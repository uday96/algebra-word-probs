from pynlp import StanfordCoreNLP
import spacy
from neuralcoref import Coref
from helper import HiddenPrints

spacy_parser = spacy.load('en')

annotators = "tokenize, ssplit, pos, lemma, ner, parse, dcoref"
options = {}

nlp = StanfordCoreNLP(annotators=annotators, options=options)

def split_parts(first_part,is_first_part):
	first_parse = spacy_parser(first_part)
	has_verb = False
	verb_str = None
	P1=""
	V1=""
	A1=""
	Pr1=""
	for token in first_parse:
		if token.pos_ == "VERB":
			has_verb = True
			verb_str = str(token.text)
			break
	if not has_verb and is_first_part:
		print("no verb found in first part")
		return P1,V1,A1,Pr1
	elif not has_verb and not is_first_part:
		after_verb = first_part
	else:
		V1 = verb_str
		P1 = first_part.split(V1+" ")[0].strip()
		after_verb = first_part.split(V1+" ")[1].strip()
	prep_parse = spacy_parser(after_verb)
	has_prep = False
	prep_str = None
	for token in prep_parse:
		if (token.tag_ == "IN" or token.tag_ == "TO") and str(token.text)!="by":
			has_prep = True
			prep_str = str(token.text)
			break
	if not has_prep:
		A1 = after_verb
	else:
		A1 = after_verb.split(prep_str+" ")[0].strip()
		Pr1 = prep_str+" "+after_verb.split(prep_str+" ")[1].strip()
	return P1,V1,A1,Pr1

def resolve_conjs(text):
	print("resolving conjunctions...")
	document = nlp(text)
	document2 = spacy_parser(text)
	has_conj = False
	conj_str = None
	conjs = ["and ","but ","if ","then "]
	for conj in conjs:
		if conj in text:
			has_conj = True
			conj_str = conj
			break
	if not has_conj:
		print("conjuction not found")
		return text,""
	parts = text.split(conj_str)
	if len(parts) != 2:
		print("Error")
	first_part = parts[0].strip()
	second_part = parts[1].strip()
	P1,V1,A1,Pr1 = split_parts(first_part,True)
	P2,V2,A2,Pr2 = split_parts(second_part,False)
	if len(P1)==0:
		P1 = P2
	if len(P2)==0:
		P2 = P1
	if len(V1)==0:
		V1 = V2
	if len(V2)==0:
		V2 = V1
	if len(A1)==0 and "to" not in Pr1:
		A1 = A2
	if len(P2)==0:
		A2 = A1
	if len(Pr1)==0:
		Pr1 = Pr2
	if len(Pr2)==0:
		Pr2 = Pr1
	first_part_new = (P1.strip()+" "+V1.strip()+" "+A1.strip()+" "+Pr1.strip()).strip()
	second_part_new = (P2.strip()+" "+V2.strip()+" "+A2.strip()+" "+Pr2.strip()).strip()
	if "." not in first_part_new:
		first_part_new += "."
	if "." not in second_part_new:
		second_part_new += "."
	return first_part_new,second_part_new

def resolve_dollars(text):
	print("resolving dollars...")
	if "$" not in text:
		return text
	parts = text.split()
	processed_text = ""
	for part in parts:
		if "$" not in part:
			processed_text += part+" "
		else:
			processed_text += part.replace("$","")+" dollars "
	return processed_text.strip()

def resolve_corefs(text):
	print("resolving pronoun co-refs...")
	document = nlp(text)
	with HiddenPrints():
		coref = Coref()
	context = ""
	for sentence in document:
		# print(str(sentence))
		if "they " in str(sentence):
			context += " "+str(sentence).strip()
			continue
		clusters = coref.one_shot_coref(utterances=str(sentence).strip(), context=context)
		resolved_utterance_text = coref.get_resolved_utterances()
		# print(resolved_utterance_text)
		context += " ".join(resolved_utterance_text).strip()
	return context

def cap_proper_nouns(text):
	document = nlp(text)
	processed_text = text
	ignore = ["the","a"]
	for sentence in document:
		for token in sentence:
			t = str(token).strip()
			if len(t) > 1 and t not in ignore:
				cap_nnp_t = t[0].upper() + t[1:]
				if cap_nnp_t in processed_text:
					processed_text = processed_text.replace(t,cap_nnp_t)
	return processed_text

def preprocess_text(text):
	document = nlp(text)
	processed_text = ""
	for sentence in document:
		first_part_new,second_part_new = resolve_conjs(str(sentence))
		if len(first_part_new)>0:
			processed_text += resolve_dollars(first_part_new)
			processed_text = processed_text.strip()
			if processed_text[-1] != '.' and processed_text[-1] != '?':
				processed_text += ' . '
		if len(second_part_new)>0:
			processed_text += resolve_dollars(second_part_new)
			if processed_text[-1] != '.' and processed_text[-1] != '?':
				processed_text += ' . '
	processed_text = processed_text.replace(" . ",".")
	processed_text = processed_text.replace(" .",".")
	processed_text = processed_text.replace(". ",".")
	processed_text = processed_text.replace("."," . ")
	processed_text = resolve_corefs(processed_text)
	processed_text = processed_text.replace(" . ",".")
	processed_text = processed_text.replace(" .",".")
	processed_text = processed_text.replace(". ",".")
	processed_text = processed_text.replace("."," . ")
	processed_text = cap_proper_nouns(processed_text)
	return processed_text


if __name__ == "__main__":
	text = "Sam had 49 pennies and 34 nickels in his bank ."
	text = 'Sara has 31 red and 15 green balloons . Sandy has 24 red balloons . How many red balloons do they have in total ? '
	text = "Mike had 34 peaches at his roadside fruit dish . He went to the orchard and picked peaches to stock up . There are now 86 peaches . how many did he pick ? "
	text = "Tom has 9 yellow balloons and Sara has 8 yellow balloons . How many yellow balloons do they have in total ? "
	# text = 'Jason found 49 seashells and 48 starfish on the beach . He gave 13 of the seashells to Tim . How many seashells does Jason now have ? '
	# text = 'Liz had 9 black kittens. She gave some of her kittens to Joan. Joan has now 11 kittens. Liz has 5 kittens left and 3 has spots. How many kittens did Joan get?'
	processed_text = preprocess_text(text)
	print(processed_text)