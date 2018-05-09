from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys

with open("../data/questions_refined.txt") as f:
    content = f.readlines()
qs = [x1.strip() for x1 in content] 

with open("../data/answers_refined.txt") as f:
    content = f.readlines()
anses = [float(x1.strip()) for x1 in content] 

count1=0
count2=0


driver = webdriver.Firefox()
driver.get('http://cogcomp.org/page/demo_view/Math')
for z in range(0,len(qs)):
	try:
		time.sleep(3)
		inputElement = driver.find_element_by_id("text")
		i2 = driver.find_element_by_xpath("//button[@href='#here']")
		inputElement.clear()
		inputElement.send_keys(qs[z])
		i2.click()
		time.sleep(5)
		ans = driver.find_element_by_id("results").text
		ans = ans.split("\n")[0].split("=")[1].strip()
		count2+=1
		if ans==str(anses[z]):
			count1+=1	
	except Exception as e:
		count2+=1
	
driver.close()
print("\n")
print count1
print count2
print count1*100/count2