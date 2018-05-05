from word_prob_solver import word_prob_solver
from helper import bcolors

def interface():
	print("\n",bcolors.OKGREEN+bcolors.BOLD+"----- ALGEBRA WORD PROBLEM SOLVER -----"+bcolors.ENDC,"\n")
	while True:
		question = input(bcolors.OKBLUE+bcolors.BOLD+"Question (or EXIT): "+bcolors.ENDC).strip()
		if question.lower() == "exit":
			print("exiting..")
			break
		print("")
		answer = word_prob_solver(question)
		# print("\n","---------------------------","\n")
		# print("Que: ",question,"\n")
		print(bcolors.WARNING+bcolors.BOLD+"\nAns: ",str(answer)+bcolors.ENDC,"\n")

if __name__ == "__main__":
	interface()