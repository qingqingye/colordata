
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from kmeans import k_mean,find_k_features
import numpy as np
import cv2
import  xdrlib ,sys
import time
import io

import requests
import os
from json import JSONDecoder
from PIL import Image
import base64

import pickle
import collections

import math
import skin_detector
# In[4]:



# def skin(H,S,V):
#     skin=0
#     if ((H >= 0) and (H <= 34 )) or ((H >= 335) and (H <= 360 )):
#             if ((S >= 0.1 * 255) and (S <= 0.6 * 255)) and (V >= 20 and V<=200)  :
#                 skin = 1
#     return skin



def get_face(photo):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key ="9aGaFpNxsPmOPq-qFc75CFJbSdGt_1lb"
    secret ="gJJ2rWcMS8dGdj-n_p2xpOQ9MaqXD6q3"
    data = {"api_key":key, "api_secret": secret, "return_attributes": None, 
            "return_landmark":0, }
    success, encoded_image = cv2.imencode('.jpg', photo)
    photo_bytes = encoded_image.tobytes()
    files = {"image_file": photo_bytes}

    response = requests.post(http_url, data=data, files=files)
    req_con = response.content.decode() #.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    result= req_dict['faces'][0]['face_rectangle']
    x, y, w, h = result["left"], result["top"], result["width"], result["height"]
    face = photo[y: y+h, x: x+w, :]
    return face, (x, y, w, h)


def preprocess(path):
    photo = cv2.imread(path)
    origin_shape = photo.shape
    photo = compress(photo, origin_shape[0], origin_shape[1], 
                     max(origin_shape[0:2])) if max(origin_shape[0:2]) > 900 else photo
    return photo


def get_skin_rgb(path, thresh=0.66):
    photo = preprocess(path)
    face, xywh = get_face(photo)
    face = cv2.resize(face, (max(1000, face.shape[0]), max(1000, face.shape[1])))
    skin_mask = skin_detector.process(face, thresh=thresh)
#     debug
    cv2.imwrite('face.jpg', face)
    cv2.imwrite('skin.jpg', cv2.bitwise_and(face, face, mask=skin_mask))
    skin_idx = skin_mask.reshape(-1) == 255.
    skins = face.reshape(-1, 3)[skin_idx.reshape(-1)]
    if skins.shape[0] == 0:
        raise ValueError 
    skin_bgr = np.mean(skins, axis=0).astype(int)
    rgb = skin_bgr[2], skin_bgr[1], skin_bgr[0]
    # print(rgb)
    return rgb





def bgr2rgb(img_bgr):
    img_rgb = np.zeros(img_bgr.shape, img_bgr.dtype)
    img_rgb[:,:,0] = img_bgr[:,:,2]
    img_rgb[:,:,1] = img_bgr[:,:,1]
    img_rgb[:,:,2] = img_bgr[:,:,0]
    return img_rgb


# In[5]:

def rgb2hex(r,g,b):
    return (r << 16) + (g << 8) + b


def visual_color(w, h, color, portion):
    # @ input:
    # h: bar height; w: bar length(width);
    # color k X 3 feature color
    # portion k X 1 each feature ratio.
    index_sorted = np.argsort(portion)    # find sorted index 
    index_sorted = index_sorted[::-1]     # apply the decrease order
    portion = portion[index_sorted]       # sort the portion in decrease order
    color_sorted = color[index_sorted, :] # sort the color in decrease order
    bar = np.zeros((w, 3))                # init bar with 1 height
    temp_len = 0
    for i in range(len(portion)):
        len_next = int(np.floor(w * portion[i] + temp_len))  # find end part of this color 
        if (i == len(portion) - 1):       # if it is the end of the color, then end color will be end of bar
            len_next = w
        for j in range(temp_len, len_next): # fill the color
            bar[j][0] = color_sorted[i][0]
            bar[j][1] = color_sorted[i][1]
            bar[j][2] = color_sorted[i][2]
        temp_len = len_next               # next color. start index.
    bar = bar.reshape(1, w, 3)            # reshape to a image.
    bar = np.tile(bar, (h, 1, 1))         # tile with height
    bar = bar.reshape(h, w, 3)            # reshape to size of h x w x 3
    bar = bar.astype(np.uint8)
    return bar , color_sorted , portion


# In[6]:


def compress(photo,h,w,_max):
    
    to_scale=900/_max
    
    if (_max==w ):
        tw=900
        th=to_scale*h
    else:
        th=900
        tw=to_scale*w
    
    newsize = (int(tw),int(th))
    # 缩放图像
    newimage = cv2.resize(photo, newsize) 
    return newimage 

    




def get_rgb(path, k=6):  
                
    photo=preprocess(path)
    shape = photo.shape
    
    http_url ="https://api-cn.faceplusplus.com/humanbodypp/beta/segment"
    key ="9aGaFpNxsPmOPq-qFc75CFJbSdGt_1lb"
    secret ="gJJ2rWcMS8dGdj-n_p2xpOQ9MaqXD6q3"

    data = {"api_key":key, "api_secret": secret, "return_attributes": "result"}
    success, encoded_image = cv2.imencode('.jpg', photo)
    photo_bytes = encoded_image.tobytes()

    files = {"image_file": photo_bytes}

    response = requests.post(http_url, data=data, files=files)
    req_con = response.content.decode() #.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    result= req_dict['result']

    segment=base64.b64decode(result)
    imggray = np.array(Image.open(io.BytesIO(segment)))
    thred = 40
    imggray[imggray<thred] = 0  #background
    imggray[imggray>=thred] = 1  #body

    # make a master after have gray already
    mask = np.stack([imggray, imggray, imggray]).transpose([1,2,0])
    newmatrix= bgr2rgb(mask*photo)

    #3 dimension to  1 dimension
    photo = newmatrix.reshape(-1,3) * 1.0

    photo_k_mean_plus, k_features, portion = k_mean(photo, k, plus=True)
    img_bar, color_sorted , portion_sorted = visual_color(1000, 50, k_features, portion)
    portion_sorted=np.array(portion_sorted).reshape(k,1)*100.0

#     output = [path, color_sorted, portion_sorted]
#     return output
    output = np.concatenate([color_sorted,portion_sorted],axis=1)
    return output , photo_k_mean_plus ,shape



def parse_feature(feature_string):
    feature_np = []
    for feature in feature_string.split("\n"):
        feature_np.append(list(map(lambda x: eval(x), feature[3:-2].split())))
    return np.array(feature_np)



# def hsv2rgb(h, s, v):
#     h = float(h)
#     s = float(s)
#     v = float(v)
#     h60 = h / 60.0
#     h60f = math.floor(h60)
#     hi = int(h60f) % 6
#     f = h60 - h60f
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
#     r, g, b = 0, 0, 0
#     if hi == 0: r, g, b = v, t, p
#     elif hi == 1: r, g, b = q, v, p
#     elif hi == 2: r, g, b = p, v, t
#     elif hi == 3: r, g, b = p, q, v
#     elif hi == 4: r, g, b = t, p, v
#     elif hi == 5: r, g, b = v, p, q
#     r, g, b = int(r * 255), int(g * 255), int(b * 255)
#     return r, g, b

def rgb2hsv(r, g, b , percent):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    s=s*100
    v=v*100
    return h, s, v , percent



def ssd(h,s,v,h1,s1,v1):
    
    hhh=np.square(float(h) - float(h1))
    vvv=np.square(float(v) - float(v1))
    sss=np.square(float(s) - float(s1))
    dis=0.8*hhh+0.1*sss+0.1*vvv
   # dis=0.8*np.square(h-h1)+0.1*np.square(s-s1)+0.1*np.square(v-v1)

    
    return dis
            
    
    

if __name__ == "__main__":
    print(type(get_skin_rgb("/home/yejj/Computer_Vision_I/K-mean_color_cluster/2018/2018ss/jueshe/2.jpg", 0.7)))
  #  print(type(get_rgb("/home/yejj/Computer_Vision_I/K-mean_color_cluster/2018/2018ss/FENGSANSAN/0.jpg", 6)))
        


# # In[64]:


# from collections import Counter

# color_dict = getColorList()
# print(color_dict)

# allphotos=[]
# for photo in artist:   #one photo of that artist
#     colorwords=[] 
    
#     count=0
#     for color in photo: 
     
        
#         get=0
#         checkdict = 0
#         for d in color_dict:
#             checkdict += 1
#             lower=color_dict[d][0]
#             upper=color_dict[d][1]
#             if (sum(color[0:3]>=lower)==3 & sum(color[0:3]<upper)==3):
               
#                 colorwords.append(d)
#                 get=100
    
#             if (checkdict == 11   and  get == 0):
#                 colorwords.append("no_related_color")
                
# #     top3 = colorwords.most_common(3)
# #     print(top3)          
#     allphotos.append(colorwords)
# for i in allphotos:   
#     print(i)


# # In[68]:


# values= sum(allphotos,[])


# from collections import Counter
 
# values_counts = Counter(values)
# top_3 = values_counts.most_common(3)
# print(top_3)


# # In[108]:


# #####################################################################
# ######### save the data##############################################
# #####################################################################
# file = open('alldata.pickle', 'wb')
# pickle.dump(allphotos, file)
# file.close()
 
# # 读取
# file = open('alldata.pickle', 'rb')
# a_dict1 = pickle.load(file)
# file.close()
# print(a_dict1)


# # In[40]:



# import json

# r = 255
# g = 125
# b = 207

# rgb_query = str(r)+","+str(g)+","+str(b)
# url_infos = "http://www.thecolorapi.com/id?rgb=" + rgb_query + "&format=name"

# r = requests.get(url = url_infos)
# json_infos = json.loads(r.text)
# name = json_infos['name']['value']
# print("the color is " + name)



# # oneline
# name = json.loads(requests.get(url = url_infos).text)['name']['value']
# print("the color is " + name)