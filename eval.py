from word_prob_solver import word_prob_solver
import json

def eval_model():
	# PATH = "./data/DfN/ALL/ALL.json"
	PATH = "./data/DfN/DS1/DS1.json"
	dataset = json.load(open(PATH,"r"))
	correct_fp = open("correct.txt","w")
	wrong_fp = open("wrong.txt","w")
	correct = 0
	count = 0
	for data in dataset:
		try:
			ans = word_prob_solver(data["sQuestion"])
			if str(ans) == str(data["lSolutions"][0]):
				correct += 1
				correct_fp.write(str(data["sQuestion"])+"\n"+str(ans)+"\n\n")
			else:
				wrong_fp.write((data["sQuestion"])+"\n"+str(e)+"\n\n")	
		except Exception as e:
			wrong_fp.write((data["sQuestion"])+"\n"+str(e)+"\n\n")
		finally :
			count += 1
			if count == 2:
				break
	correct_fp.close()
	wrong_fp.close()
	print("correct: ",str(correct))
	print("total: ",str(len(dataset)))
	print("acc: ",str(float(correct)/len(dataset)))

if __name__ == "__main__":
	eval_model()