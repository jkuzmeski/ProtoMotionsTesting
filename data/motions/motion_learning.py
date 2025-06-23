# the goal of the script is to open a npy file that contains a motion and understand what data is in it
import numpy as np


data = np.load('D:\\Isaac\\ProtoMotions\\data\\motions\\smpl_humanoid_walk.npy', allow_pickle=True)

# the data is a ndarray can you help me understand what is in each element of the ndarray and how i can recreate it with my own data
print(data.shape)  # Print the shape of the array to understand its dimensions
print(data.dtype)  # Print the data type of the elements in the array
print(data)  # Print the entire array to see its contents

# To understand the contents better, you can iterate through the array
for i, element in enumerate(data):
    print(f"Element {i}: {element}")
