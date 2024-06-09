#!/usr/bin/env python
# coding: utf-8

# In[20]:


import random as random

experimentnumber = 100

def montyhall(experimentnumber):
    
    successChange = 0
    successNoChange = 0
    
    for i in range(experimentnumber):
        
        doors = [0,0,0]
        car_door = random.randint(0,2)
        
        contestant_choice = random.randint(0, 2)
        
        monty_choices = [door for door in range(0, 3) if door != contestant_choice and door != car_door]
        monty_open = random.choice(monty_choices)
        
        switch_choice = [door for door in range(0, 3) if door != contestant_choice and door != monty_open][0]
        if switch_choice == car_door:
            successChange += 1
            
        if contestant_choice == car_door:
            successNoChange += 1      
    
    return successChange, successNoChange


successChange, successNoChange = montyhall(experimentnumber)

print("The success rate with changing the door is: ", successChange, " the success rate without changing the door is: ", successNoChange)
    


# In[ ]:





# In[ ]:




