from word_prob_solver import word_prob_solver
import json
from helper import HiddenPrints

def eval_model_from_file():
	with open("../data/questions_refined.txt") as f:
	    content = f.readlines()
	qs = [x1.strip() for x1 in content] 
	with open("../data/answers_refined.txt") as f:
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
