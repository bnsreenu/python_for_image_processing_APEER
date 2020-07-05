#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Use logical conditions for if and else statements
"""

#If and else statements 

a = 10
b = 10

if a < b:
    print("Hey, a is smaller than b")  #indent code within if statement
    
else:
    print("Hmm, looks like a is greater than b") #indent code within else statement
    

#USe elif for multiple conditions
    
if a < b:
    print("Hey, a is smaller than b")  #indent code within if statement
    
elif a == b:
    print("Wow, both numbers are the same")
    
else:
    print("Hmm, looks like a is greater than b") #indent code within else statement

#Use and / or for conditions
    
a = 10
b = 5
c = 20

if a > b and b > c:
    print("Here a is > b and b > c")
    
elif a > b or b > c:
    print("Either a > b or b < c")
    
#Conditions can be nested
if a > b:
    print("First if statement where a > b")
    if b < c:
        print("Second level statement where b < c")
    elif b > c:
        print("Second level statement where b > c")
elif a <= b:
    print("Well, a is less than b")