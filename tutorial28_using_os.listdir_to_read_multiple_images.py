#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
os.listdir:
returns a list containing the names of the entries in the directory given by path

"""
import os

path = 'images/test_images/'
print(os.listdir(path))  #Very similar to glob, prints a list of all files in the directory

for image in os.listdir(path):  #iterate through each file to perform some action
    print(image)
    
#############################
"""
#os.walk --     
returns a generator, that creates a tuple of values 
(current_path, directories in current_path, files in current_path).   

Every time the generator is called it will follow each directory recursively 
until no further sub-directories are available from the initial directory 
that walk was called upon.

os.path.join() method in Python join one or more path components intelligently.
""" 
import os
print(os.walk("."))  #Nothing to see here as this is just a generator object

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("."):
    #print(root)  #Prints root directory names
    
    path = root.split(os.sep)  #SPlit at separator (/ or \)
    #print(path)  #Gives names of directories for easy location of files
    #print(files)   #Prints all file names in all directories
    
#Let us now visualize directories and files within them
    print((len(path) - 1) * '---', os.path.basename(root)) #Add --- based on the path
    for file in files:
        print(len(path) * '---', file)
        
 #######################     
#Another way to look at all dirs. and files...  
import os
for root, dirs, files in os.walk("."):
#for path,subdir,files in os.walk("."):
   for name in dirs:
       print (os.path.join(root, name)) # will print path of directories
   for name in files:    
       print (os.path.join(root, name)) # will print path of files        