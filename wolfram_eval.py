import requests
import urllib

with open("questions_refined.txt") as f:
    content = f.readlines()
qs = [x1.strip() for x1 in content] 

with open("aa.txt") as f:
    content = f.readlines()
anses = [x1.strip() for x1 in content] 

count1=0
count2=0

for z in range(0,len(qs)):
	print z
	inp = qs[z]
	f = { 'input' : inp , 'format' : 'plaintext' , 'output' : 'JSON' , 'appid' : 'RLE7A9-PRT46WWGE7'}
	f = urllib.urlencode(f)
	url = 'https://api.wolframalpha.com/v2/query?'+f

	response = requests.get(url)
	try:
		ans = response.json()["queryresult"]["pods"][1]["subpods"][0]["plaintext"]
		x = [int(s) for s in ans.split() if s.isdigit()]
		count2+=1
		if str(x[0])==anses[z]:
			count1+=1	
	except Exception as e:
		count2+=1
	
print count1
print count2
print count1*100/count2