#!/usr/bin/env python
# coding: utf-8

# # Prewitt

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


img = cv2.imread('C:\\Users\\rabia\\Downloads/3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(gray, (3,3), 0)


# In[15]:


kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])


# In[16]:


img_prewittx = cv2.filter2D(img_gray, -1, kernelx)
img_prewitty = cv2.filter2D(img_gray, -1, kernely)


# In[17]:


plt.subplot(2,2,1), plt.imshow(img, cmap = "gray")
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.imshow(img_prewittx, cmap = "gray")
plt.title('PrewittX'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3), plt.imshow(img_prewitty, cmap = "gray")
plt.title('PrewittY'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4), plt.imshow(img_prewittx+img_prewitty, cmap = "gray")
plt.title('PrewittXY'), plt.xticks([]), plt.yticks([])


# In[18]:


# plt.show()


# # Histogram
# 

# In[19]:


from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure


# In[20]:


imshow(img)


# In[21]:


MC=True


# In[22]:


hogfv, hog_image = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2),
                      visualize=True, multichannel=MC)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,5))

imshow(hog_image_rescaled)


# In[ ]:




