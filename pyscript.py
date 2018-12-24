import os
import evelib as el
import json
from tqdm import tqdm
import cv2
import numpy as np
root = "/home/yejj/Computer_Vision_I/K-mean_color_cluster/2019_woman"

fail_list = []

redo = True
#initialize 
show_="2019_woman"
if not os.path.exists("results/" + show_+"_meta"):
    os.makedirs("results/" + show_ +"_meta")
path_meta = os.path.join("results", show_+"_meta")
meta_list = [] if redo else os.listdir(path_meta)

for show in os.listdir(root):        #get the show name
    if not (os.path.isdir(os.path.join(root, show))):
        print(os.path.join(root, show))
        print(os.path.isdir(os.path.join(root, show)))
        continue
    # show operation
    print("=> begin show, "+ show)
    brand_list = os.listdir(os.path.join(root, show))   #get the brand
    show_gallery = {}
    show_gallery["show_name"] = show
    brand_infos = []
    for brand in brand_list:
        print(brand)
        brand_path = os.path.join(root, show, brand)
        brands = {}
        brands["brand_name"] = brand
        
        if brand in meta_list:
            print("\t|_" + brand + " already done. skip")
            brands = json.load(open(os.path.join(path_meta, brand), 'r'))
            brand_infos.append(brands)
            continue
        else:
            print("\t|_" + brand + " doing")
        # brand operation
        if not os.path.isdir(brand_path):
            continue
            
        single_brand = []
        for pic in tqdm(os.listdir(brand_path)):
            if pic[0:2] == "._":
                continue
            pic_json = {}
            pic_json["image"] = pic
            pic_path = os.path.join(brand_path, pic)
            path_key = os.path.join("2019", show, brand, pic)
            pic_json["path"] = path_key
            try:
                rgbarray = el.get_rgb(pic_path, 6)
                skin_rgb = el.get_skin_rgb(pic_path, 0.5)
                top6_rgb = rgbarray[0].tolist()
                dis=[]
                for colors in top6_rgb:
                    distance=0
                    for i in range(0,3):
                        distance+=(colors[i]-float(skin_rgb[i]))**2
                    dis.append(distance)

                skin_mum=np.argsort(dis)[0]
                del top6_rgb[skin_mum]
                pic_json["features"] = top6_rgb
            except KeyboardInterrupt:
                exit(-1)
            except:
                fail_list.append(pic_path)
                continue
                print("\t\t|_" + pic)
 
            ###########save diff###########################
            photo_k_mean_plus= rgbarray[1]
            shape=rgbarray[2]
            photo_ok = el.bgr2rgb (photo_k_mean_plus.reshape(shape) )
            newpic=pic.split(".")[0]+"_kmeans.tiff"
            cv2.imwrite(os.path.join("/home/yejj/Computer_Vision_I/K-mean_color_scluster/picresult",newpic), photo_ok)
            ################################################  
            single_brand.append(pic_json)   #EXPAND single brand and push to the dic

        brands["pic_list"] = single_brand
  ################### create data and save at the same time to prevent disturbing  and losing data ###########################
        brand_save=str(show)+"_"+str(brand)
        with open(os.path.join(path_meta, brand_save), "w") as me:
            json.dump(brands, me, indent=2, separators=(',', ': ')  )


        brand_infos.append(brands)                  # a show contains some brands
    show_gallery["brands"] = brand_infos

    ##########################  make a  final json #################### 
    with open("results/" + show+"_new.json", 'w') as f:
        json.dump(show_gallery, f, indent=2, separators=(',', ': '))
    with open("results/" + show+"_faillist.txt", 'w') as fail:
        fail.write(str(fail_list))
    print("show :" + show + "DONE.")
