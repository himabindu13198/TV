
# coding: utf-8

# In[1]:


import sys

from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average


# In[2]:


def compare_images(img1, img2):
    # normalize to compensate for exposure difference
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    sum1 = img1 + img2  # Sum
    z_norm = sum(sum1)
    return (m_norm, z_norm)


# In[3]:


def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr


# In[4]:


def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng


# In[5]:


# print("hey")
file1 = "test3_pre.png"
file2 = "test3_post.png"
# read images as 2D arrays (convert to grayscale for simplicity)
img1 = to_grayscale(imread(file1).astype(float))
img2 = to_grayscale(imread(file2).astype(float))
# compare
n_m, n_0 = compare_images(img1, img2)
print("Pixel by pixel difference:", n_m/img1.size)
print("Initial building pixels", (n_m + n_0)/(2*img1.size))

