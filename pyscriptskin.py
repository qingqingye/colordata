import os
import evelib as el
import json
import numpy as np
from tqdm import tqdm
root = "/home/yejj/Computer_Vision_I/K-mean_color_cluster/2019"

fail_list = []

restart = True
#initialize 
show_ = "2019"
if not os.path.exists("skinresults/" + show_+"_meta"):
    os.makedirs("skinresults/" + show_ +"_meta")
path_meta = os.path.join("skinresults", show_+"_meta")
meta_list = [] if restart else os.listdir(path_meta)


for show in os.listdir(root):        #get the show name
    if not (os.path.isdir(os.path.join(root, show)) ):
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
        # print(brand)
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
            #try:
            pic_json["skin_rgb"] = str(el.get_skin_rgb(pic_path, 0.5))#np.array(el.get_skin_rgb(pic_path, 0.8)).tolist()
            print(pic_json,"111111111111111111111")    
            # except KeyboardInterrupt:
            #     exit(-1)
            # except:
            #     fail_list.append(pic_path)
            #     continue
#                     print("\t\t|_" + pic)
            single_brand.append(pic_json)   #EXPAND single brand and push to the dic
            print(single_brand,"222222222222")
        brands["pic_list"] = single_brand
        ####################### create data and save at the same time to prevent disturbing  and losing data ###########################
        with open(os.path.join(path_meta, brand), "w") as me:
            json.dump(brands, me, indent=2, separators=(',', ': ')  )


        brand_infos.append(brands)                  # a show contains some brands
    show_gallery["brands"] = brand_infos

    ##########################  make a  final json #################### 
    with open("skinresults/" + show+"_new.json", 'w') as f:
        json.dump(show_gallery, f, indent=2, separators=(',', ': '))
    with open("skinresults/" + show+"_faillist.txt", 'w') as fail:
        fail.write(str(fail_list))
    print("show :" + show + "DONE.")
