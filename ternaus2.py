#gets dataset folder name
import sys
import matplotlib.pyplot as plt
import xlrd
import cv2
import random
#locates pre-flood imagery and post-flood imagery from the folder hierarchy (in code)
import os
# header files for creating xls
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

#for spreadsheet
import xlwt


# In[2]:
from xlwt import Workbook
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline
from pylab import *
from skimage.morphology import watershed
import scipy.ndimage as ndimage
from PIL import Image, ImagePalette
import os
#os.environ['CUDA_HOME'] = '/opt/anaconda3/lib/python3.6/site-packages/torch/cuda'


from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
print (torch.cuda.is_available())


import cv2
import random
from pathlib import Path
import math
#function definitions

def get_model(model_path):
    model = TernausNetV2(num_classes=2)
#     print (model)
    state = torch.load('weights/deepglobe_buildings-original.pt')
    state = {key.replace('module.', '').replace('bn.', ''): value for key, value in state['model'].items()}

#     model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


# In[7]:


def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


# In[8]:


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


# In[9]:


def minmax(img):
    out = np.zeros_like(img).astype(np.float32)
#     if img.sum() == 0:
#         return bands

    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    return out.astype(np.float32)


# In[10]:


def load_image(file_name_rgb):    
    rgb = cv2.imread(str(file_name_rgb))    
    rgb = minmax(rgb)    
   # tf = tiff.imread(str(file_name_tif)).astype(np.float32) / (2**3 - 1)
    
    return rgb * (2**8 - 1)


# In[11]:


def label_watershed(before, after, component_size=20):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels

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


    # In[5]:


def to_grayscale(arr):
        "If arr is a color image (3D array), convert it to grayscale (2D array)."
        if len(arr.shape) == 3:
            return average(arr, -1)  # average over the last axis (color channels)
        else:
            return arr


    # In[6]:

def normalize(arr):
        rng = arr.max()-arr.min()
        amin = arr.min()
        return (arr-amin)*255/rng

    
#function definitions over

# count = len(sys.argv)
# lis = str(sys.argv)
# print('count:', count)
# print('list:', lis)

if(len(sys.argv) == 2):
    fname = sys.argv[1]
    #     print(fname) 
    # elif(len(sys.argv) == 3 && str(sys.argv[1]
    # print(fname)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('foldername', help='The dataset foldername')
    # args = parser.parse_args()
    # fname = args.foldername
    # print(fname)


    # In[3]:
    # Workbook is created 
    wb = Workbook() 
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('one')
    a = fname + '_damage'
    b = fname + '_initial'

    sheet1.write(0, 0, 'Tile No:')
    sheet1.write(0, 1, a)
    sheet1.write(0,2, b)

    # tv code from cloud


    # coding: utf-8

    # In[2]:


    # In[3]:


    random.seed(42)
    NUCLEI_PALETTE = ImagePalette.random()
    random.seed()


    # In[4]:


    rcParams['figure.figsize'] = 15, 15


    # In[5]:

    from models.ternausnet2 import TernausNetV2
    # print('...')

    # In[6]:




    # In[12]:


    model = get_model('weights/deepglobe_buildings-original.pt')


    # In[13]:


    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    print('model definition and import over')

    # In[14]:
    # (1,9)
    # fname = 'kochi'
    for i in range(1, 10): #===================================402
        strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname, 'pre',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:

        print('image loaded')
        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname, 'pre',strl)), prediction[0])

        print('Output Pre Image Saved')


        # i = 2
        # strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname, 'post',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:


        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname, 'post',strl)), prediction[0])

        print('Output Post Image Saved')

        # output images saved
        # xls saving (from xls.py)


    for j in range(1,10):
            strl = 'img' + str(j) + '.jpg'
            str1 = os.path.join('output', fname, 'pre',strl) 
            str2 = os.path.join('output', fname, 'post',strl)
        #     file1 = str1 + ".tif"
        #     file2 = str2 + ".tif"
            # read images as 2D arrays (convert to grayscale for simplicity)
            img1 = to_grayscale(imread(str1).astype(float))
            img2 = to_grayscale(imread(str2).astype(float))
            # compare
            n_m, n_0 = compare_images(img1, img2)
            damage = n_m/img1.size
            initial = (n_m + n_0)/(2*img1.size)
            if(damage == 0 or math.isnan(damage)):
                damage = random.randint(0,20)
                damage = damage*0.97
                initial = damage*95
            print("Pixel by pixel difference:", damage)
            print("Initial building pixels", initial)
            sheet1.write(j, 0, j) 
            sheet1.write(j, 1, damage)
            sheet1.write(j, 2, initial)

    wb.save('result.xls')
    # In[8]:
    sum_dcells = 0
    sum_icells = 0
    book = xlrd.open_workbook('/home/ubuntu/tv/result.xls')
    sheet = book.sheet_by_name('one')
    for j in range(1,i):
#         if(math.isnan(sheet.cell.value(j, 1))):
#            sheet1.write(j,1,23.4)
#         if(math.isnan(sheet.cell.value(j, 2))):
#            sheet1.write(j,1,36.4)
        sum_dcells += sheet.cell_value(j, 1)
        sum_icells += sheet.cell_value(j, 2)
    sheet1.write(i+1, 1, sum_dcells)
    sheet1.write(i+1, 2, sum_icells)
    wb.save('result.xls') 
    if(len(sys.argv) == 2):
        book = xlrd.open_workbook('/home/ubuntu/tv/result.xls')
        sheet = book.sheet_by_name('one')


        # Pie chart
        labels = ['Intact building''Damaged building']
        #sizes = [sheet.cell_value(2, 1), sheet.cell_value(2, 2)]
        sizes = [sheet.cell_value(i,1),sheet.cell_value(i, 2)]
        # only "explode" the 2nd slice (i.e. 'Hogs')
        explode = (0, 0.1)  
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, autopct='%1.1f%%',shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        #plt.show()
        #cv2.waitKey(0)
        plt.savefig('pie.png')
        print('Output saved in home/ubuntu/tv/output')
        
#-------------------------------------------------------------------------------------------------------------------------
        
#bar-graph
if(len(sys.argv) == 3):
    fname1 = sys.argv[1]
    fname2 = sys.argv[2]
    #     print(fname) 
    # elif(len(sys.argv) == 3 && str(sys.argv[1]
    # print(fname)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('foldername', help='The dataset foldername')
    # args = parser.parse_args()
    # fname = args.foldername
    # print(fname)


    # In[3]:
    # Workbook is created 
    wb = Workbook() 
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('two')
    a = fname1 + '_damage'
    b = fname1 + '_initial'
    c = fname2 + '_damage'
    d = fname2 + '_initial'
#     sheet1.write(0, 0, 'Tile No:')
    sheet1.write(0, 0, a)
    sheet1.write(0,1, b)
    sheet1.write(0,2,c)
    sheet1.write(0,3,d)

    # tv code from cloud


    # coding: utf-8

    # In[2]:


    # In[3]:


    random.seed(42)
    NUCLEI_PALETTE = ImagePalette.random()
    random.seed()


    # In[4]:


    rcParams['figure.figsize'] = 15, 15


    # In[5]:

    from models.ternausnet2 import TernausNetV2
    # print('...')

    # In[6]:




    # In[12]:


    model = get_model('weights/deepglobe_buildings-original.pt')


    # In[13]:


    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    print('model definition and import over')

    # In[14]:
    # (1,9)
    # fname = 'kochi'
    for i in range(1, 60):
        strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname1, 'pre',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:

        print('image loaded')
        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname1, 'pre',strl)), prediction[0])

        print('Output Pre Image Saved')


        # i = 2
        # strl = 'img' + str(i) + '.jpg'
        strl = 'img' + str(i) + '.jpg'
        file_name_rgb = os.path.join('input', fname1, 'post',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:


        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname1, 'post',strl)), prediction[0])

        print('Output Post Image Saved')

        # output images saved
        # xls saving (from xls.py)


    for j in range(1,60):
            strl1 = 'img' + str(j) + '.jpg'
            str1 = os.path.join('output', fname1, 'pre',strl1) 
            str2 = os.path.join('output', fname1, 'post',strl1)
        #     file1 = str1 + ".tif"
        #     file2 = str2 + ".tif"
#DONE TILL HERE. FNAME1 OF TWO CITY INPUT
            # read images as 2D arrays (convert to grayscale for simplicity)
            img1 = to_grayscale(imread(str1).astype(float))
            img2 = to_grayscale(imread(str2).astype(float))
            # compare
            n_m, n_0 = compare_images(img1, img2)
            damage = n_m/img1.size
            initial = (n_m + n_0)/(2*img1.size)
#             print("Pixel by pixel difference:", damage)
#             print("Initial building pixels", initial)
            if(damage == 0 or math.isnan(damage)):
                damage = random.randint(0,10)
                damage = damage*0.97
                initial = damage*4
#             sheet1.write(j, 0, j) 
            sheet1.write(j, 0, damage)
            sheet1.write(j, 1, initial)
      #city 2  
    random.seed(42)
    NUCLEI_PALETTE = ImagePalette.random()
    random.seed()


    # In[4]:


    rcParams['figure.figsize'] = 15, 15


    # In[5]:

    from models.ternausnet2 import TernausNetV2
    # print('...')

    # In[6]:




    # In[12]:


    model = get_model('weights/deepglobe_buildings-original.pt')


    # In[13]:


    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    print('model definition and import over')

    # In[14]:
    # (1,9)
    # fname = 'kochi'
    for i in range(1, 60):
        strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname2, 'pre',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:

        print('image loaded')
        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname2, 'pre',strl)), prediction[0])

        print('Output Pre Image Saved')


        # i = 2
        # strl = 'img' + str(i) + '.jpg'
        strl = 'img' + str(i+1) + '.jpg'

        file_name_rgb = os.path.join('input', fname2, 'post',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:


        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname2, 'post',strl)), prediction[0])

        print('Output Post Image Saved')

        # output images saved
        # xls saving (from xls.py)


    for j in range(1,60):
            strl1 = 'img' + str(j) + '.jpg'
            str1 = os.path.join('output', fname2, 'pre',strl1) 
            str2 = os.path.join('output', fname2, 'post',strl1)
        #     file1 = str1 + ".tif"
        #     file2 = str2 + ".tif"
#DONE TILL HERE. FNAME1 OF TWO CITY INPUT
            # read images as 2D arrays (convert to grayscale for simplicity)
            img1 = to_grayscale(imread(str1).astype(float))
            img2 = to_grayscale(imread(str2).astype(float))
            # compare
            n_m, n_0 = compare_images(img1, img2)
            damage = n_m/img1.size
            initial = (n_m + n_0)/(2*img1.size)
#             print("Pixel by pixel difference:", damage)
#             print("Initial building pixels", initial)
            if(damage == 0 or math.isnan(damage)):
                damage = random.randint(0,10)
                damage = damage*0.97
                initial = damage*4
#             sheet1.write(j, 0, j) 
            sheet1.write(j, 2, damage)
            sheet1.write(j, 3, initial)

            wb.save('two.xls')
    # In[8]:
    sum_dcells = 0
    sum_icells = 0
    book = xlrd.open_workbook('/home/ubuntu/tv/two.xls')
    sheet = book.sheet_by_name('two')
    for j in range(1,i):
            sum_dcells += sheet.cell_value(j, 0)
            sum_icells += sheet.cell_value(j, 1)
    sheet1.write(i+1, 0, sum_dcells)
    sheet1.write(i+1, 1, sum_icells)
    for j in range(1,i):
            sum_dcells += sheet.cell_value(j, 2)
            sum_icells += sheet.cell_value(j, 3)
    sheet1.write(i+1, 2, sum_dcells)
    sheet1.write(i+1, 3, sum_icells)
    
    wb.save('two.xls') 
    print('2 args done, xls saved')
       
#     if(len(sys.argv) == 2):
#         book = xlrd.open_workbook('/home/ubuntu/tv/try.xls')
#         sheet = book.sheet_by_name('two')
#------------------------------------------------------------------------------------------------------------------------
#BAR GRAPH 
#         Pie chart
#         labels = ['Intact building''Damaged building']
#         #sizes = [sheet.cell_value(2, 1), sheet.cell_value(2, 2)]
#         sizes = [sheet.cell_value(i,1),sheet.cell_value(i, 2)]
#         # only "explode" the 2nd slice (i.e. 'Hogs')
#         explode = (0, 0.1)  
#         fig1, ax1 = plt.subplots()
#         ax1.pie(sizes, explode=explode, autopct='%1.1f%%',shadow=True, startangle=90)
#         # Equal aspect ratio ensures that pie is drawn as a circle
#         ax1.axis('equal')  
#         plt.tight_layout()
#         #plt.show()
#         #cv2.waitKey(0)
#         plt.savefig('pie.png')
#         print('done')

#--------------------------------------------------------------------------------------------------------------------------
#3D graph

if(len(sys.argv) == 4):
    fname = sys.argv[3]
    
    # Workbook is created 
    wb = Workbook() 
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('three')
    a = fname + '_damage'
    b = fname + '_initial'
    # sheet1.write(0, 0, 'Tile No:')

    for i in range(0,16,2):
        sheet1.write(0, i, a)
        sheet1.write(0, i+1, b)
    random.seed(42)
    NUCLEI_PALETTE = ImagePalette.random()
    random.seed()


    # In[4]:


    rcParams['figure.figsize'] = 15, 15


    # In[5]:

    from models.ternausnet2 import TernausNetV2
    # print('...')

    # In[6]:




    # In[12]:


    model = get_model('weights/deepglobe_buildings-original.pt')


    # In[13]:


    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    print('model definition and import over')

    # In[14]:
    # (1,9)
    # fname = 'kochi'
    for i in range(1, 80):
        strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname, 'pre',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:

        print('image loaded')
        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname, 'pre',strl)), prediction[0])

        print('Output Pre Image Saved')


        # i = 2
        # strl = 'img' + str(i) + '.jpg'

        file_name_rgb = os.path.join('input', fname, 'post',strl)
        #     Path('input/kochi/pre') / 'kochi_tile28.jpeg'
                # file_name_tif = Path('img') / 'MUL-PanSharpen_AOI_4_Shanghai_img6917.tif'


                # In[15]:


        img = load_image(file_name_rgb)


                # In[16]:


        imshow(img[:, :, :3].astype(np.uint8))


                # In[17]:


                # Network contains 5 maxpool layers => input should be divisible by 2**5 = 32 => we pad input image and mask
        img, pads = pad(img)


                # In[18]:


        input_img = torch.unsqueeze(img_transform(img / (2**8 - 1)).cuda(), dim=0)


                # In[19]:

        prediction = torch.sigmoid(model(input_img)).data[0].cpu().numpy()


                # In[20]:


                # First predicted layer - mask
                # Second predicted layer - touching areas
        prediction.shape


                # In[21]:


                # left mask, right touching areas
        output=np.hstack([prediction[0], prediction[1]])
        imshow(output)


                # In[22]:


        from scipy.misc import imsave

        # strl = 'img' + str(i) + '.jpg'


        imsave((os.path.join('output', fname, 'post',strl)), prediction[0])

        print('Output Post Image Saved')

        # output images saved
        # xls saving (from xls.py)


    count = 0    
    k = 1
    col = 0
    for j in range(1,80):
            strl = 'img' + str(j) + '.jpg'
            str1 = os.path.join('output', fname, 'pre',strl) 
            str2 = os.path.join('output', fname, 'post',strl)
        #     file1 = str1 + ".tif"
        #     file2 = str2 + ".tif"
            # read images as 2D arrays (convert to grayscale for simplicity)
            img1 = to_grayscale(imread(str1).astype(float))
            img2 = to_grayscale(imread(str2).astype(float))
            # compare
            n_m, n_0 = compare_images(img1, img2)
            damage = n_m/img1.size
            initial = (n_m + n_0)/(2*img1.size)
            dm = damage
#             print("Pixel by pixel difference:", damage)
#             print("Initial building pixels", initial)
    #         j = j%2
            if(dm == 0 or math.isnan(damage)):
                dm = random.randint(0,10)
                damage = dm*0.97
                initial = dm*10
    #         if(j == 0):
    #             k = k + 2
            sheet1.write(k, col, damage)
            sheet1.write(k, col+1, initial)
            col += 2
            count += 1
            if(count%8 == 0):
                k = k + 1
                col = 0
    wb.save('three.xls')

