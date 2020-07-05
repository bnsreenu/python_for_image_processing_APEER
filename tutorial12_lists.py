#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Lists: 
    Set of numbers or text.

"""

a = [1,2,3,4,5]  #Set of numbers
b = [2, 4, 'apple', 'banana'] #Numbers and/or text

print(type(a))

print(b[2])  #3rd element in the list

b.append('grape')  #add to the list. Lists are editable
print(b)

c = b.pop()  #Removes the last entry and assigns to c
print(b)
print(c)

d = list(range(20))  #Generate a list using range
print(d)

e = d[3:7]  #Subset of d from 4th to 7th element.
print(e)

f = d[11:]  #Subset from 12th element to end of the list
print(f)

g = d[:5]  #Subset upto 5th element
print(g)

h = a+b  #Add lists, concatenate
print(h)
print(len(h)) #tells the length of the list

print(max(d))

#Multidimensional lists

x = [[1,2], [3,4]]
print(2*x)

#Multiplying every element by 2 is not easy with lists.
#Just repeats the list that many times.
#This is where we can rely on the strength of numpy.