#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
For loops

"""

#For loops: Execute a set of statements, once for each item in a list, string, tuples, etc. 

a = [1,2,3,4,5]

for i in a:
    print("Current value is: ", i)


for x in "microscope":
    print(x)

#Break and continue statements can be used to exit or continue.. similar to while

microscopes = ["confocal", "widefield", "fluorescence"]

for x in microscopes:
    print(x)
    if x == "widefield":
        break  
    

### using the range function 
#range function returns a sequence of numbers. 
        
for i in range(10):
    print(i)    
    
for i in range(20, 60, 2):
    print(i)
    
for num in range(0, 20):
    if num%2 ==0:
        print("%d is an even number" %(num))
    else:
        print("%d is an odd number" %(num))