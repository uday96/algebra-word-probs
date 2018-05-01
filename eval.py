from word_prob_solver import word_prob_solver
import json

def eval_model():
	PATH = "./data/DfN/ALL/ALL.json"
	dataset = json.load(open(PATH,"r"))
	correct_fp = open("correct.txt","w")
	wrong_fp = open("wrong.txt","w")
	correct = 0
	for data in dataset:
		try:
			ans = word_prob_solver(data["sQuestion"])
			if ans == data["lSolutions"][0]:
				correct += 1
				correct_fp.write(data["sQuestion"]+"\n"+ans+"\n\n")
		except Exception as e:
			wrong_fp.write(data["sQuestion"]+"\n"+str(e)+"\n\n")
	print("correct: ",str(correct))
	print("total: ",str(len(data)))
	print("acc: ",str(float(correct)/len(data)))

if __name__ == "__main__":
	eval_model()