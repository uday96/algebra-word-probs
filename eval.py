from word_prob_solver import word_prob_solver
import json
from helper import HiddenPrints

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
				wrong_fp.write(str(data["sQuestion"])+"\n"+str(anses)+"\n\n")	
		except Exception as e:
			wrong_fp.write(str(data["sQuestion"])+"\n"+str(e)+"\n\n")
		finally :
			count += 1
			if count == 2:
				break
	correct_fp.close()
	wrong_fp.close()
	print("correct: ",str(correct))
	print("total: ",str(len(dataset)))
	print("acc: ",str(float(correct)/len(dataset)))

def eval_model_from_file():
	with open("questions_refined.txt") as f:
	    content = f.readlines()
	qs = [x1.strip() for x1 in content] 
	with open("answers_refined.txt") as f:
	    content = f.readlines()
	anses = [x1.strip() for x1 in content] 
	correct_fp = open("correct.txt","w")
	wrong_fp = open("wrong.txt","w")
	count1=0
	count2=0
	for z in range(0,len(qs)):
		print(z)
		inp = qs[z]
		try:
			with HiddenPrints():
				ans = word_prob_solver(inp)
			count2+=1
			if str(ans)==anses[z]:
				count1+=1
				correct_fp.write(str(inp)+"\n"+str(ans)+"\n\n")
			else:
				wrong_fp.write(str(inp)+"\n"+str(ans)+"\n\n")	
		except Exception as e:
			count2+=1
			wrong_fp.write(str(inp)+"\n"+str(e)+"\n\n")	
	print("correct: ",str(count1))
	print("total: ",str(count2))
	print("acc: ",str((float(count1)*100)/count2))

if __name__ == "__main__":
	eval_model_from_file()
	# correct:  66
	# total:  141
	# acc:  46.808510638297875
