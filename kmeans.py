import matplotlib.pyplot as plt
import numpy as np
import cv2



def bgr2rgb(img_bgr):
    img_rgb = np.zeros(img_bgr.shape, img_bgr.dtype)
    img_rgb[:,:,0] = img_bgr[:,:,2]
    img_rgb[:,:,1] = img_bgr[:,:,1]
    img_rgb[:,:,2] = img_bgr[:,:,0]
    return img_rgb

def find_k_features(photo,k):
    """
    k-mean ++, input is an array
    """
    
    l = photo.shape[0]
    k_feature = photo[np.random.choice(l,1)]
    fixed_norm = np.sum(np.square(photo),axis=1,keepdims=True)  # N x 3
    for i in range(k-1):
        k_square_sum = np.sum(np.square(k_feature),axis=1,keepdims=False) # M x 3
        distance = ((fixed_norm + k_square_sum).T - 2*np.dot(k_feature,np.transpose(photo))) # M x N
        P = np.cumsum(np.min(distance,axis=0)/np.sum(np.min(distance,axis=0)))
        seed = np.random.random()
        new_k = photo[P>seed][0]
        k_feature = np.vstack((k_feature,new_k))
    return k_feature

def k_mean(photo, k, plus =True):
    """
    k-means algorithm, plus is optional
    """
#     for i in photo:
#         if (not (photo[i][0]==0.0 and photo[i][1] == 0.0 and photo[i][2] == 0.0)):
#             print(photo)
#     print(photo.shape)
#     photo = photo[np.nonzero(photo)]
#     print(photo.shape)
    k = k+1
    if (plus == True):
        # Use the function to find best k_features
        # k means++: Find the k longest point
        
        k_feature = find_k_features(photo,k)
       
    else: 
        # find the feature pixel vector RANDOMLY.
        k_feature = photo[np.random.choice(photo.shape[0],k,replace=False)]
        
    last_loss = 0
    fixed_norm = np.sum(np.square(photo),axis=1,keepdims=True)  # N x 3
    # while k center still changing
    for i in range(1000):
        
        k_square_sum = np.sum(np.square(k_feature),axis=1,keepdims=False) # M x 3
        distance = ((fixed_norm + k_square_sum).T- 2*np.dot(k_feature,np.transpose(photo))) # M x N
        
        ############################################################
        #   Vectorize progress, aspired from CS231n assignment 1 KNN Part.
        #   , makes this faster! =)
        ##########################################################
        min_loss = np.sum(np.min(distance,axis=0))
        if (abs(last_loss-min_loss) /min_loss < 0.01):
            # check if we converge.
            break
            
        last_loss = min_loss
        min_index = np.argmin(distance,axis = 0)

        # update each k feature to average.
        for i in range(k):
            k_feature[i] = np.average(photo[min_index == i],axis =0)
            
    # make picture with k colors
    count = []
    k_feature = np.floor(k_feature)
    index_0 = -1;
    for i in range(k):
        if ((k_feature[i][0] == 0 and k_feature[i][1] == 0 and k_feature[i][2] == 0)  or  (k_feature[i][0] + k_feature[i][1] + k_feature[i][2]) <5):
            index_0 = i
        photo[min_index == i] = k_feature[i]
        count.append(np.sum(photo == k_feature[i]))
    k_feature = np.delete(k_feature, index_0 ,0)
    count.remove(count[index_0])
    count = np.array(count)
    portion = count / np.sum(count)
    return photo, k_feature, portion

def main():
    k = 5
    print("=> select k = %d" % k)

    ########        K_MEANS_PLUSPLUS      #########
    photo = cv2.imread("bird_large.tiff")
    origin_shape = photo.shape
    photo = photo.reshape(-1,3) * 1.0
    photo_k_mean_plus = k_mean(photo,k,plus=True)
    print("=> kmeans++ complete, check \"after_k_plus.tiff\" ")
    photo_ok = photo_k_mean_plus.reshape(origin_shape)
    k_photo = cv2.imwrite('after_k_plus.tiff',photo_ok)
    ########        K_MEAN_CLASSIC      #########
    photo = cv2.imread("bird_large.tiff")
    origin_shape = photo.shape
    photo = photo.reshape(-1,3) * 1.0
    photo_k_mean = k_mean(photo,k,plus=False)
    print("=> kmeans classical complete")
    photo_ok2 = photo_k_mean.reshape(origin_shape)
    k_photo = cv2.imwrite('after_k_classic.tiff',photo_ok2)

    
    
if __name__ == "__main__":
    main()
    
