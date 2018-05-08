from pynlp import StanfordCoreNLP
import spacy
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
from preprocess import preprocess_text
import re, json
from copy import deepcopy

def chunk_tree_to_sent(tree, concat=' '):
	punct_re = re.compile(r'\s([,\.;\?])')
	s = concat.join([t for t in tree.leaves()])
	return re.sub(punct_re, r'\g<1>', s)

def ExtractNounPhrases( myTree, phrase="NP"):
	myPhrases = []
	if (myTree.label() == phrase):
		has_child_NP = False
		for child in myTree:
			if(child.label() == phrase):
				has_child_NP = True
				break
		if not has_child_NP:
			myPhrases.append( myTree.copy(True) )
	for child in myTree:
		if (type(child) is Tree):
			list_of_phrases = ExtractNounPhrases(child, phrase)
			if (len(list_of_phrases) > 0):
				myPhrases.extend(list_of_phrases)
	return myPhrases

def get_numbers(document):
	print("identifying numbers...")
	sentences = []
	numbers = []
	for index, sentence in enumerate(document):
		a = [[str(entity),str(sentence),index] for entity in sentence.entities if str(entity.type) == 'NUMBER']
		numbers.extend(a)
		sentences.append(str(sentence))
	if len(numbers) == 0:
		for sid in range(len(sentences)):
			sentence = sentences[sid]
			for token in sentence.split():
				if check_int(token):
					numbers.append([token,sentence,sid])
	return sentences,numbers

def get_num_dep_nouns(document,dep_parser):
	print("identifying nouns dep numbers...")
	h = set([])
	for sentence in document:
		result = dep_parser.raw_parse(str(sentence))
		for parse in result:
			for dep in list(parse.triples()):
				if dep[1] == "nummod":
					h.update([str(dep[0][0])])
	h = list(h)
	return h

def filter_et(et,sentences,numbers):
	et_filtered = []
	for sentence in sentences:
		sent_ets = []
		for et_x in et:
			if et_x[2] == sentence:
				sent_ets.append(et_x)
		if len(sent_ets) > 1:
			picked_et = False
			for sent_et in sent_ets:
				for num in numbers:
					if num[0] in sent_et[0] and sent_et[2] == num[1]:
						picked_et = True
						et_filtered.append(sent_et)
						break
				if picked_et:
					break
		elif len(sent_ets) == 1:
			et_filtered.append(sent_ets[0])
	return et_filtered

def get_noun_phrases_entities(h,sentences,tree_parser):
	print("identifying NPs & et...")
	NPs=[]
	et=[]
	parse_trees = sum([list(dep_graphs) for dep_graphs in tree_parser.raw_parse_sents(sentences)], []) 
	for sid in range(len(parse_trees)):
		tree = parse_trees[sid]
		NP_tree_list = ExtractNounPhrases(tree, 'NP')
		for np in NP_tree_list:
			np_str = chunk_tree_to_sent(np)
			NPs.append([np_str,sentences[sid]])
			for h_noun in h:
				if h_noun in np_str:
					et.append([np_str,h_noun,sentences[sid],sid])
	return NPs,et

def get_nearest_verb(verb_set,et):
	sentence = et[2]
	entity = et[0]
	nearest_verb = "$"
	min_dist = 10000
	e_ind = sentence.find(entity)
	for verb in verb_set:
		v_ind = sentence.find(verb)
		dist = abs(e_ind-v_ind)
		if dist < min_dist:
			nearest_verb = verb
			min_dist = dist
	return nearest_verb

def get_verbs(et,non_lem_sents,dep_parser,nlp):
	print("identifying vt...")
	vt=[]
	for entity in et:
		verb_set = set([])
		document = nlp(non_lem_sents[entity[3]])
		for token in document[0]:
			if "VB" in token.pos:
				verb_set.update([str(token)])
		result = dep_parser.raw_parse(non_lem_sents[entity[3]])
		for parse in result:
			for dep in list(parse.triples()):
				if "VB" in dep[0][1] and dep[0][1] != "VBN":
					verb_set.update([dep[0][0]])
				if "VB" in dep[2][1] and dep[0][1] != "VBN":
					verb_set.update([dep[2][0]])
		verb_set = list(verb_set)
		if len(verb_set) > 1 and "did" in verb_set:
			verb_set.remove("did")
		if len(verb_set) > 1 and "does" in verb_set:
			verb_set.remove("does")
		if len(verb_set) > 1 and "do" in verb_set:
			verb_set.remove("do")
		if len(verb_set) > 1 and "were" in verb_set:
			verb_set.remove("were")
		nearest_verb = get_nearest_verb(verb_set,entity)
		vt.append([nearest_verb,entity[1],entity[2]])
	return vt

def get_ex(document2,sentences,h):
	print("identifying ex...")
	ex = None
	for np in document2.noun_chunks:
		if(np.root.text in h):
			ex = [np.text,np.root.text,sentences[-1],len(sentences)-1]
	return ex

def get_numt(et,numbers):
	print("identifying numt...")
	numt = []
	variable_dcount = 0
	variable = "$"+str(variable_dcount)
	for chunk in et:
		flag = False
		for num in numbers:
			if num[0] in chunk[0] and chunk[0] in num[1]:
				flag = True
				numt.append([num[0],chunk[1],chunk[2]])
				numbers.remove(num)
				break
		if not flag:
			numt.append([variable,chunk[1],chunk[2]])
			variable_dcount += 1
			variable = "$"+str(variable_dcount)
	return numt

def process_bare_num(numbers,sentences,numt,et,h):
	if len(numbers)==0:
		return
	print("bare numbers found...")
	for num in numbers:
		h_noun_back = None
		h_noun_front = None
		back_ind = 1000
		front_ind = 1000
		dist = 0	#backward
		parts = num[1].split()
		nind = parts.index(num[0])
		flag = False
		for i in range(nind-1,-1,-1):
			dist += 1
			if parts[i] in h:
				flag = True
				h_noun_back = parts[i]
				break
		if not flag:
			for j in range(num[2]-1,-1,-1):
				parts = str(sentences[j]).split()
				for i in range(len(parts)-1,-1,-1):
					dist += 1
					if parts[i] in h:
						flag = True
						h_noun_back = parts[i]
						break
				if flag:
					break
		back_ind = dist
		dist = 0	#forward
		parts = num[1].split()
		nind = parts.index(num[0])
		flag = False
		for i in range(nind+1,len(parts),1):
			dist += 1
			if parts[i] in h:
				flag = True
				h_noun_front = parts[i]
				break
		if not flag:
			for j in range(num[2]+1,len(sentences),1):
				parts = str(sentences[j]).split()
				for i in range(0,len(parts),1):
					dist += 1
					if parts[i] in h:
						flag = True
						h_noun_front = parts[i]
						break
				if flag:
					break
		front_ind = dist
		h_noun_num = None
		if h_noun_back and h_noun_front:
			if front_ind < back_ind:
				h_noun_num = h_noun_front
			else:
				h_noun_num = h_noun_back
		elif h_noun_back:
			h_noun_num = h_noun_back
		elif h_noun_front:
			h_noun_num = h_noun_front
		if h_noun_num:
			# print([num[0],h_noun_num])
			update_existing_numt = False
			for enum in numt:
				if("$" in enum[0] and enum[1] == h_noun_num and enum[2] == num[1]):
					update_existing_numt = True
					enum[0] = num[0]
					break
			if not update_existing_numt:
				numt.append([num[0],h_noun_num,num[1]])
				et.append([num[0]+" "+h_noun_num,h_noun_num,num[1],num[2]])
				# vt.append([num[0]+"|"+h_noun_num,h_noun_num,num[1]])
		else:
			print("No entity found for bare number: "+str(num[0]))

def get_attributes(et,ex,dep_parser):
	print("identifying ats...")
	at = []
	last_at = "$"
	for e in et+[ex]:
		result = dep_parser.raw_parse(str(e[0]))
		flag = False
		dep_tup = None
		for parse in result:
			for dep in list(parse.triples()):
				if dep[1] == "amod" and dep[0][0]==e[1] and dep[2][0] != "many":
					flag = True
					at.append([str(dep[2][0]),e[1],e[2]])
					last_at = str(dep[2][0])
					break
				elif dep[1] == "dep" and dep[2][0]==e[1] and dep[0][0] != "many":
					dep_tup = dep
		if not flag and dep_tup:
			flag = True
			at.append([str(dep_tup[0][0]),e[1],e[2]])
			last_at = str(dep_tup[0][0])
		if not flag:
			# print("No attribute found for entity: "+str(e[1]))
			at.append([last_at,e[1],e[2]])
	print("identifying ax...")
	ax = at[-1]
	del at[-1]
	return at,ax

def get_fragments(et,numt,vt,at,ex,vx,ax,sentences,non_lem_sents):
	print("identifying fragments...")
	fragments = []
	for sid in range(len(sentences)-1):
		has_frag = False
		for e,n,v,a  in zip(et,numt,vt,at):
			if sentences[sid] == e[2]:
				fragments.append([non_lem_sents[sid],e[1],e[0],n[0],v[0],a[0]])
				has_frag = True
		if not has_frag:
			fragments.append([non_lem_sents[sid],"$","$","$","$","$"])
	fragments.append([non_lem_sents[-1],ex[1],ex[0],"$",vx[0],ax[0]])
	return fragments

def get_containers(fragments,h,dep_parser,nlp):
	print("identifying ct...")
	ct=[]
	last_ct="$"
	is_loc_related = False
	loc_ct = None
	ignore = ["how","now","total","all"]
	for fragment in fragments:
		result = dep_parser.raw_parse(fragment[0])
		ct1="$"
		ct2="$"
		bool_there_is = False
		if "there is" in fragment[0].lower() or "there are" in fragment[0].lower() or "there will be" in fragment[0].lower():
			bool_there_is = True
		for parse in result:
			for dep in list(parse.triples()):
				if dep[1] == "nsubj" and dep[0][0] == fragment[4] and not bool_there_is:
					if ct1 == "$" and dep[2][0] not in h and dep[2][0].lower() not in ignore:
						ct1 = dep[2][0]
						last_ct = ct1
				elif dep[1] == "nmod:poss" and dep[0][0] == ct1 and not bool_there_is:
					if dep[2][0] not in h:
						ct1 = dep[2][0]
						last_ct = ct1
				elif dep[1] == "case" and dep[2][0] == "in" and bool_there_is:
					if ct1 == "$" and dep[0][0] not in h:
						ct1 = dep[0][0]
						last_ct = ct1
						loc_ct = ct1
						is_loc_related = True
				elif dep[1] == "case" and dep[2][0] == "in" and is_loc_related and "how many" in fragment[0].lower():
					if ct1 == "$" and dep[0][0] not in h and dep[0][0].lower() not in ignore:
						ct1 = dep[0][0]
						last_ct = ct1
					elif ct1 == "$" and loc_ct:
						ct1 = loc_ct
						last_ct = ct1
				elif dep[1] == "case" and dep[2][0] == "by":
					if ct1 == "$" and dep[0][0] not in h:
						ct1 = dep[0][0]
						last_ct = ct1
					elif ct2 == "$" and dep[0][0] not in h:
						ct2 = dep[0][0]
						# last_ct = ct2
				elif dep[1] == "compound" and dep[0][0] == ct1:
					if ct2 == "$" and dep[2][0] not in h:
						ct2 = dep[2][0]
						# last_ct = ct2
				elif dep[1] == "iobj" and dep[0][0] == fragment[4]:
					if ct2 == "$" and dep[2][0] not in h:
						ct2 = dep[2][0]
						# last_ct = ct2
				elif dep[1] == "nmod" and dep[0][0] == fragment[4]:
					if ct2 == "$" and dep[2][0] not in h:
						ct2 = dep[2][0]
						# last_ct = ct2
				elif dep[1] == "advmod" and dep[0][0] == fragment[4]:
					if ct2 == "$" and dep[2][0] not in h and dep[2][0].lower() not in ignore:
						ct2 = dep[2][0]
						# last_ct = ct2
				elif dep[1] == "xcomp" and dep[0][0] == fragment[4]:
					if ct2 == "$" and dep[2][0] not in h:
						ct2 = dep[2][0]
						# last_ct = ct2
				elif dep[1] == "nsubjpass" and is_loc_related and "how many" in fragment[0].lower():
					if ct2 == "$" and dep[2][0] not in h:
						ct2 = dep[2][0]
						# last_ct = ct2
		if ct1 == "$" and ct2 != "$":
			ct1 = ct2
		if ct2 == "$" and ct1 != "$":
			ct2 = ct1
		if ct1 == "$":
			ct1 = last_ct
		if ct2 == "$":
			ct2 = last_ct
		dct = nlp(ct1+" "+ct2)
		ct1 = dct[0][0].lemma.lower()
		ct2 = dct[0][1].lemma.lower()
		ct.append([ct1,ct2,fragment[0]])
		fragment.append([ct1,ct2])
	return ct

#Sentence Verb Categorization
def verb_category(verb,nlp,verb_cats_json):
	verb = verb.lower().strip()
	dv = nlp(verb)
	verb_lem = dv[0][0].lemma.lower()
	if verb_lem in verb_cats_json:
		return verb_cats_json[verb_lem]
	else:
		print("Error: Verb Category not found")
		return None

def get_states(fragments,verb_cats,ex,ax):
	#State Progression
	print("building states...")
	states = []
	initialiser = "J0"
	for fragment,vcat in zip(fragments,verb_cats):
		state={}
		if len(states) > 0:
			state = deepcopy(states[-1])
		if fragment[1] == ex[1] and fragment[5] == ax[0]:
			if vcat == "OBS":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":fragment[3],"E":fragment[1],"A":fragment[5]}]
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] = fragment[3]
							break
			elif vcat == "NEG":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"-"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "-"+fragment[3]
							break
			elif vcat == "POS":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"+"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "+"+fragment[3]
							break
			elif vcat == "NEG_TR":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"-"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "-"+fragment[3]
							break
				if fragment[6][1].lower() != fragment[6][0].lower():
					if fragment[6][1].lower() not in state:
						state[fragment[6][1].lower()] = [{"N":initialiser+"+"+fragment[3],"E":fragment[1],"A":fragment[5]}]
						initialiser += "0"
					else:
						ct_state_ets = state[fragment[6][1].lower()]
						for ct_et in ct_state_ets:
							if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
								ct_et["N"] += "+"+fragment[3]
								break
			elif vcat == "POS_TR":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"+"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "+"+fragment[3]
							break
				if fragment[6][1].lower() != fragment[6][0].lower():
					if fragment[6][1].lower() not in state:
						state[fragment[6][1].lower()] = [{"N":initialiser+"-"+fragment[3],"E":fragment[1],"A":fragment[5]}]
						initialiser += "0"
					else:
						ct_state_ets = state[fragment[6][1].lower()]
						for ct_et in ct_state_ets:
							if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
								ct_et["N"] += "-"+fragment[3]
								break
			elif vcat == "DESTROY":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"-"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "-"+fragment[3]
							break
				if fragment[6][1].lower() != fragment[6][0].lower():
					if fragment[6][1].lower() not in state:
						state[fragment[6][1].lower()] = [{"N":initialiser+"-"+fragment[3],"E":fragment[1],"A":fragment[5]}]
						initialiser += "0"
					else:
						ct_state_ets = state[fragment[6][1].lower()]
						for ct_et in ct_state_ets:
							if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
								ct_et["N"] += "-"+fragment[3]
								break
			elif vcat == "CONSTRUCT":
				if fragment[6][0].lower() not in state:
					state[fragment[6][0].lower()] = [{"N":initialiser+"+"+fragment[3],"E":fragment[1],"A":fragment[5]}]
					initialiser += "0"
				else:
					ct_state_ets = state[fragment[6][0].lower()]
					for ct_et in ct_state_ets:
						if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
							ct_et["N"] += "+"+fragment[3]
							break
				if fragment[6][1].lower() != fragment[6][0].lower():
					if fragment[6][1].lower() not in state:
						state[fragment[6][1].lower()] = [{"N":initialiser+"+"+fragment[3],"E":fragment[1],"A":fragment[5]}]
						initialiser += "0"
					else:
						ct_state_ets = state[fragment[6][1].lower()]
						for ct_et in ct_state_ets:
							if ct_et["E"] == fragment[1] and ct_et["A"] == fragment[5]:
								ct_et["N"] += "+"+fragment[3]
								break
		states.append(state)
	print("final states: ","\n")
	for state in states:
		print(state,"\n")
	return states

def init_parsers():
	print("initializing parsers...")
	spacy_parser = spacy.load('en')
	path_to_jar = './stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar'
	path_to_models_jar = './stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar'
	dep_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
	tree_parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
	annotators = "tokenize, ssplit, pos, lemma, ner, parse, dcoref"
	options = {}
	nlp = StanfordCoreNLP(annotators=annotators, options=options)
	verb_cats_json = json.load(open("verb_cats.json","r"))
	return nlp,spacy_parser,dep_parser,tree_parser,verb_cats_json

def build_equations(states):
	print("building equations...")
	eq_subs = {}
	equations = []
	for state in states:
		for ct in state:
			ct_state_ets = state[ct]
			for ct_et in ct_state_ets:
				eq_subs_str = ct+"_"+ct_et["E"]+"_"+ct_et["A"]
				if eq_subs_str not in eq_subs:
					eq_subs[eq_subs_str] = [ct_et["N"]]
				else:
					eq_subs[eq_subs_str].append(ct_et["N"])
	for eq in eq_subs:
		# print(eq," : ",eq_subs[eq])
		for eqi in range(len(eq_subs[eq])):
			if eqi+1 < len(eq_subs[eq]):
				if "$" in eq_subs[eq][eqi] and "$" not in eq_subs[eq][eqi+1]:
					equation = eq_subs[eq][eqi]+"="+eq_subs[eq][eqi+1]
					equations.append(equation)
					# print(equation)

	print("\n",equations,"\n")
	return equations

def check_int(s):
	s = s.strip()
	if s[0] in ['-', '+']:
		return s[1:].isdigit()
	return s.isdigit()

def solve(equations):
	solutions = {}
	for eq in equations:
		parts = eq.split("=")
		if "+" in parts[0]:
			lhs = parts[0].strip().split("+")
			if "$" in lhs[0] and check_int(lhs[1]) and check_int(parts[1]):
				solutions[lhs[0]] = str(int(parts[1].strip()) -  int(lhs[1].strip()))
			elif "$" in lhs[1] and check_int(lhs[0]) and check_int(parts[1]):
				solutions[lhs[1]] = str(int(parts[1].strip()) -  int(lhs[0].strip()))
		elif "-" in parts[0]:
			lhs = parts[0].strip().split("-")
			if "$" in lhs[0] and check_int(lhs[1]) and check_int(parts[1]):
				solutions[lhs[0]] = str(int(parts[1].strip()) +  int(lhs[1].strip()))
			elif "$" in lhs[1] and check_int(lhs[0]) and check_int(parts[1]):
				solutions[lhs[1]] = str(int(lhs[0].strip()) -  int(parts[1].strip()))
	return solutions

def solve_equations(equations):
	# assuming lhs always has variable and rhs has int
	print("solving equations...")
	solutions = solve(equations)
	# print("\n",solutions,"\n")
	if len(solutions) == 0:
		rep_init_eqs = []
		for eq in equations:
			rep_init_eqs.append(set_inits_zero(eq))
		solutions = solve(rep_init_eqs)
		equations = rep_init_eqs
		# print(solutions)
		# print(equations,"\n")
	for eq in equations:
		for sol in solutions:
			if sol in eq:
				eq = eq.replace(sol,solutions[sol])
		# print(eq)
		parts = eq.split("=")
		if "+" in parts[0]:
			lhs = parts[0].strip().split("+")
			if "J0" in lhs[0] and check_int(lhs[1]) and check_int(parts[1]):
				solutions[lhs[0]] = str(int(parts[1].strip()) -  int(lhs[1].strip()))
			elif "J0" in lhs[1] and check_int(lhs[0]) and check_int(parts[1]):
				solutions[lhs[1]] = str(int(parts[1].strip()) -  int(lhs[0].strip()))
		elif "-" in parts[0]:
			lhs = parts[0].strip().split("-")
			if "J0" in lhs[0] and check_int(lhs[1]) and check_int(parts[1]):
				solutions[lhs[0]] = str(int(parts[1].strip()) +  int(lhs[1].strip()))
			elif "J0" in lhs[1] and check_int(lhs[0]) and check_int(parts[1]):
				solutions[lhs[1]] = str(int(lhs[0].strip()) -  int(parts[1].strip()))
	print("\n",solutions,"\n")
	return solutions

def bool_opposite_verbs(v1_cat,v2_cat):
	if v1_cat is None or v2_cat is None:
		return False
	if "NEG" in v1_cat and "POS" in v2_cat:
		return True
	elif "POS" in v1_cat and "NEG" in v2_cat:
		return True
	return False

def set_inits_zero(ans):
	if "J" not in ans:
		return ans
	while "J" in ans:
		j_token = ""
		j_ind = ans.index("J")
		for ch in ans[j_ind:]:
			j_token += ch
			if ch in ["+","-","="]:
				break
		ans = ans.replace(j_token,"")
	return ans

def get_answer(solutions,states,fragments,fragx,orig_text,nlp,verb_cats_json):
	print("getting answer...")
	ctx1,ctx2 = fragx[6]
	ex = fragx[1]
	ax = fragx[5]
	vx = fragx[4]
	dvx = nlp(vx)
	vx_lem = dvx[0][0].lemma
	vx_cat = verb_category(vx,nlp,verb_cats_json)
	ans = ""
	bool_get_init = False
	bool_they_total = False
	NNPs = set([])
	combines = ["total","together","all"]
	if "start" in fragx[0].lower() or "begin" in fragx[0].lower():
		bool_get_init = True
	if ctx1 == "they" and (ctx2 in combines):
		bool_they_total = True
		ddp = nlp(orig_text)
		for dds in ddp:
			for ddt in dds:
				if ddt.pos == "NNP":
					NNPs.update([str(ddt.lemma).lower()])
	NNPs = list(NNPs)
	if vx_cat == "OBS" and ctx1.lower() in states[-1]:
		if bool_get_init:
			for state_x in states:
				if ans:
					break
				if not bool_they_total:
					for state in state_x[ctx1.lower()]:
						if state["E"] == ex and state["A"] == ax:
							ans = state["N"]
							break
				else:
					for nnp in NNPs:
						for state in state_x[nnp.lower()]:
							if state["E"] == ex and state["A"] == ax:
								ans += "+"+state["N"]
								break
		else:
			if not bool_they_total:
				for state in states[-1][ctx1.lower()]:
					if state["E"] == ex and state["A"] == ax:
						ans = state["N"]
						break
			else:
				for nnp in NNPs:
					for state in states[-1][nnp.lower()]:
						if state["E"] == ex and state["A"] == ax:
							ans += "+"+state["N"]
							break
	else:
		for fragment in fragments:
			vt = fragment[4]
			dvt = nlp(vt)
			vt_lem = dvt[0][0].lemma
			if not bool_they_total:
				if vx_cat == "POS" or vx_cat == "NEG":
					if ctx1 == fragment[6][0] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5]:
						ans += "+"+fragment[3]
				else:
					if ctx1 != ctx2:
						if ctx1 == fragment[6][0] and ctx2 == fragment[6][1] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5]:
							ans += "+"+fragment[3]
					else:
						if ctx1 == fragment[6][0] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5]:
							ans += "+"+fragment[3]
						elif ctx1 != fragment[6][0] and ctx2 == fragment[6][1] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5] and vx_cat in ["CONSTRUCT","DESTROY"]:
							ans += "+"+fragment[3]
			else:
				for nnp in NNPs:
					if nnp == fragment[6][0] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5]:
						ans += "+"+fragment[3]
					elif nnp != fragment[6][0] and nnp == fragment[6][1] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5] and vx_cat in ["CONSTRUCT","DESTROY"]:
						ans += "+"+fragment[3]
	if not ans:
		for fragment in fragments:
			vt = fragment[4]
			dvt = nlp(vt)
			vt_lem = dvt[0][0].lemma
			vt_cat = verb_category(vt,nlp,verb_cats_json)
			if vt_cat is None:
				continue
			opposite_verbs = bool_opposite_verbs(vx_cat,vt_cat)
			if opposite_verbs:
				if ctx1 != ctx2:
					if ctx1 == fragment[6][1] and ctx2 == fragment[6][0] and ex == fragment[1] and ax == fragment[5]:
						ans += "+"+fragment[3]
				else:
					if ctx1 == fragment[6][1] and ex == fragment[1] and ax == fragment[5]:
						ans += "+"+fragment[3]
					elif ctx1 != fragment[6][0] and ctx2 == fragment[6][1] and vx_lem == vt_lem and ex == fragment[1] and ax == fragment[5] and vx_cat in ["CONSTRUCT","DESTROY"]:
						ans += "+"+fragment[3]
	if ans:
		if ans[0] == '+':
			ans = ans[1:]
		for sol in solutions:
			if sol in ans:
				ans = ans.replace(sol,solutions[sol])
		ans = ans.strip()
		ans = set_inits_zero(ans)
		try:
			ans = eval(ans)
		except Exception as e:
			pass
	# print("Ans: ",ans)
	return ans

def word_prob_solver(text):
	orig_text = text
	nlp,spacy_parser,dep_parser,tree_parser,verb_cats_json = init_parsers()
	text = preprocess_text(text)
	# print(text)
	document = nlp(text)
	non_lem_sents = [str(sent) for sent in document]
	h = get_num_dep_nouns(document,dep_parser)
	h_lem = set([])
	is_h_lemmatized = False
	for h_noun in h:
		dh = nlp(h_noun)
		h_noun_lem = dh[0][0].lemma
		if h_noun_lem != h_noun:
			text = text.replace(h_noun,h_noun_lem)
			is_h_lemmatized = True
		h_lem.update([h_noun_lem])
	all_h = deepcopy(h)
	h = list(h_lem)
	if is_h_lemmatized:
		document = nlp(text)
		# print(text)
	sentences,numbers = get_numbers(document)
	NPs,et = get_noun_phrases_entities(h,sentences,tree_parser)
	et = filter_et(et,sentences,numbers)
	document2 = spacy_parser(text)
	ex = get_ex(document2,sentences,h)
	numt = get_numt(et,numbers)
	process_bare_num(numbers,sentences,numt,et,h)
	vt = get_verbs(et+[ex],non_lem_sents,dep_parser,nlp)
	vx = vt[-1]
	del vt[-1]
	at,ax = get_attributes(et,ex,dep_parser)
	fragments = get_fragments(et,numt,vt,at,ex,vx,ax,sentences,non_lem_sents)
	assert len(fragments) == len(sentences)
	ct = get_containers(fragments,all_h,dep_parser,nlp)
	print("final fragments :","\n")
	for fragment in fragments:
		print(fragment,"\n")
	fragx = fragments[-1]
	del fragments[-1]
	verb_cats = []
	for fragment in fragments:
		verb_cats.append(verb_category(fragment[4],nlp,verb_cats_json))
	states = get_states(fragments,verb_cats,ex,ax)
	equations = build_equations(states)
	solutions = solve_equations(equations)
	answer = get_answer(solutions,states,fragments,fragx,orig_text,nlp,verb_cats_json)
	return answer

if __name__ == "__main__":
	text = "Fred has 40 baseball cards . Keith bought 22 baseball cards from Fred . How many baseball cards does Fred have now ? "
	# TODO
	# text = 'Mary is baking a cake . The recipe wants 8 cups of flour . She already put in 2 cups . How many cups does she need to add ? '
	# TODO - conj
	# text = "Jason has 43 blue and 16 red marbles . Tom has 24 blue marbles . How many blue marbles do they have in all ? "
	# text = 'Sara has 31 red and 15 green balloons . Sandy has 24 red balloons . How many red balloons do they have in total ? '
	# TODO - coref prob
	# text = 'There were 28 bales of hay in the barn . Tim stacked bales in the barn today . There are now 54 bales of hay in the barn . How many bales did he store in the barn ? '
	# text = "Tim 's cat had kittens . He gave 3 to Jessica and 6 to Sara . He now has 9 kittens . How many kittens did he have to start with ?"
	# text = "Jason had 49 quarters in his bank . His dad gave him 25 quarters . How many quarters does he have now ? "
	# TODO - ct map
	# text = "Sara 's high school played 12 basketball games this year . The team won most of their games . They were defeated during 4 games . How many games did they win ? "
	# TODO - during
	# text = "A restaurant served 9 pizzas during lunch and 6 during dinner today . How many pizzas were served today during lunch? "
	# TODO - total
	# text = "A restaurant served 5 cakes during lunch and 6 during dinner today . The restaurant served 3 cakes yesterday . How many cakes were served in total ? "
	# text = "Melanie picked 4 plums . Dan picked 9 plums and Sally picked 3 plums from the plum tree . How many plums were picked in total ?"
	# text = "Sara picked 45 pears and Sally picked 11 pears from the pear tree . How many pears were picked in total ? "
	# text = "Joan picked 37 oranges and Sara picked 10 oranges . Alyssa picked 30 pears . How many oranges were picked in total ? "
	# text = "Benny picked 2 apples and Dan picked 9 apples from the apple tree . How many apples were picked in total ? "
	# TODO - preprocess
	# text = "There are 7 dogwood trees currently in the park . Park workers will plant 3 dogwood trees today and 2 dogwood trees tomorrow . How many dogwood trees will the park have when the workers are finished ? "
	# text = "Fred went to 36 basketball games this year but missed 35 games. He went to 11 games last year . How many basketball games did Fred go to in total ? "
	# TODO - num & coref
	# text = "Jason had 49 quarters in his bank . His dad gave him 25 quarters . How many quarters does he have now ? "
	# text = "Sam had 49 pennies and 24 nickels in his bank . His dad gave him 39 nickels and 31 quarters . How many nickels does he have now ?"
	# TODO - verb sim
	# text = "Jason went to 11 football games this month . He went to 17 games last month and plans to go to 16 games next month . How many games will he attend in all ? "
	# TODO - amod
	# text = "Mike has 35 books in his library . He bought several books at a yard sale over the weekend . He now has 56 books in his library . How many books did he buy at the yard sale ? "
	# TODO - currency and conj
	# text = "Joan purchased a basketball game for $ 5.20 , and a racing game for $ 4.23 . How much did Joan spend on video games ? "
	# TODO - unable to ignore noise
	# text = "Joan decided to sell all of her old books . She gathered up 33 books to sell . She sold 26 books in a yard sale . How many books does Joan now have ? "
	# TODO - ct detection
	# text = "Fred has 40 baseball cards . Keith bought 22 of Fred's baseball cards . How many baseball cards does Fred have now ? "
	answer = word_prob_solver(text)
	print("\n","---------------------------","\n")
	print("Que: ",text,"\n")
	print("Ans: ",answer,"\n")