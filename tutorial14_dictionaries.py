#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Dictionary is also a collection of objects like list
Dictionary's objects are key-value pairs.
Maps key to associated value

Just like lists dictionaries are mutable, can be edited.

"""

life_sciences = {'Botany':'plants',
                 'Zoology':'animals',
                 'Virology':'viruses',
                 'Cell_biology': 'cells'}

#DIctionaries can also be built using dict() function

life_sciences = dict([('Botany','plants'),
                 ('Zoology','animals'),
                 ('Virology','viruses'),
                 ('Cell_biology', 'cells')])

#Another way if key values are simple strings

life_sciences = dict(Botany='plants',
                 Zoology='animals',
                 Virology='viruses',
                 Cell_biology= 'cells')


print(life_sciences)

print('Zoology' in life_sciences)  #Returns True

#Accessing values

print(life_sciences['Botany'])

#Adding an entry
life_sciences['Neuroscience'] = 'nervous_system'

#Delete a key
del life_sciences['Neuroscience']

#Dictionaries can have combination of numbers and text for values and keys

a = {42: 'hello', 1.23: 'there', True: 'hi'}

#Dictionary keys can also be tuples as they are immutable

b = {(1, 0): 'a', (1, 1): 'b', (2, 2): 'c', (3, 2): 'd'}

#Lists cannot be keys. Following should not work. 
c = {[1, 0]: 'a', [1, 1]: 'b', [2, 2]: 'c', [3, 2]: 'd'}

b.clear()  #Clears the dictionary
print(b)

d = list(life_sciences.keys())  #Save keys as a list
e = list(life_sciences.values())
print(d)

