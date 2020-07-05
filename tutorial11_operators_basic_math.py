#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

#This tutorial explains operators and basic math

a = 2 #assign a variable

a == 2 #logical operator

b=5

a<b

#Why does this matter?
if a<b:
    print("Wow, coding is easy!")
    
else:
    print('I hope I can code one day')
    
#Simple math equation: E=mc^2

#External input:
m=input("Please enter mass: ")
m=int(m)
#m = 10 #kg
c = 300000000 #m/s

E = m*(c**2)
print("Energy=", E)
print("Energy = ", '{:.2E}'.format(E))