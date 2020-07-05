#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""

while: do something while a condition holds true

for: do things for multiple times , especially in lists or strings
    
for and while loops can also be nested.

    
    """

### WHile
    
count = 0

while count<10:
    print("The count is:", count)
    count = count + 1
    
#Be warned of infinite loop...

count = 0

while count==0:
    print("Oh my god I got stuck")
    
#You can use break statement even if the while condition is true
    
count = 0

while count<10:
    print("The count is:", count)
    if count ==5:
        break
    count = count + 1

#Continue: USe continue to stop current iteration, continue with next.
    #Here if count = 5 it skips (continues) and moves on. 
count = 0

while count<10:
    count = count + 1
    if count ==5:
        continue
    print("The count is:", count)
    



## Let's automate Centigrade to F conversion for some values
    #from -40 to 40 with steps of 5

#F = (9/5)C + 32   

C = -40

while C <= 40:
    F = ((9./5)*C) + 32
    print(C, "C in Fahrenheit is: ",F, "F")
    C = C + 5
    
    
####################################
