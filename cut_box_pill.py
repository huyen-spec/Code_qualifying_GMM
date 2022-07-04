import pandas as pd
import os
from PIL import Image, ExifTags
import json
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break
# data_dir = '/home/huyen/projects/huypn/FewShotDetection/data/vaipe/train/images/'
# annotation_dir = '/home/huyen/projects/huypn/FewShotDetection/data/vaipe/train/labels/'

data_dir = '/home/huyen/projects/huypn/FewShotDetection/data/vaipe/test/images/'
annotation_dir = '/home/huyen/projects/huypn/FewShotDetection/data/vaipe/test/labels/'

def Flipped(img):
    s = False
    h = 0
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s=True
            h=270
        elif rotation == 8:
            s = True
            h=90
    except:
            pass
    return s, h
# def exif_size(file_name):
#     img = Image.open(data_dir + file_name)
#     s = img.size

#     try:
#         rotation = dict(img._getexif().items())[orientation]
#         if rotation == 6:
#             s = (s[1], s[0])
#         elif rotation == 8:
#             s = (s[1], s[0])
#     except:
#         pass
#     return s


def run():
    label_list = []
    df_cols = ['Dir','x','y','h','w','Label']
    new_df = pd.DataFrame(columns=df_cols)
    for file_name in os.listdir(data_dir):
        # img, ori_size = read_img(file_name)
        annotation_name = file_name.split('.')[0] + '.json'
        img_name = file_name.split('.')[0]
        bboxes_json = open(os.path.join(annotation_dir, annotation_name))
        bboxes = json.load(bboxes_json)['boxes']

        for box in bboxes:
    
            if box['label'] not in label_list:
                label_list.append(box['label'])
                new_class_dir = 'pill_img_by_class/' + box['label']
                if not os.path.exists(new_class_dir):
                    os.makedirs(new_class_dir)
                    new_file_name = new_class_dir + '/' + str(img_name) +'.jpg'

            new_file_name = 'pill_img_by_class/' + box['label'] + '/' + str(img_name) +'.jpg'

            # s = exif_size(file_name)
            # # if ori_size[0] != s[0]:
            #     temp = box['w']
            #     box['w'] = box['h']
            #     box['h'] = temp
            #     temp = box['x']
            #     box['x'] = box['y']
            #     box['y'] = temp

            coords = (box['x'],    box['y'],    box['x']+ box['w'],     box['y']+box['h'])
            # print(img_name, box, coords)
            cropped_image = crop(data_dir+file_name, coords, new_file_name)

            # crop = img.crop(box)ropped_image.save(new_class_file_name , 'x': box['x'], 'y': box['y'], 'h': box['h'], 'w': box['w'], 'Label': box['label']}, ignore_index=True)
            # pd: 1: dir, id, bbox,

# do statistic on a dataframe
def statistic(new_df): # -> 564 x 564
    print(new_df.head())
    # plt.hist(new_df['h'])
    # plt.title("Height distribution")
    # plt.show()
    # plt.hist(new_df['w'])
    # plt.title("Width distribution")
    # plt.show()

    print('Mean height',new_df['h'].mean())
    print('Mean width',new_df['w'].mean())

    # print('The samples with height = 520',len(new_df[new_df['h'] == 520]))

    print(new_df.h.value_counts()[:10])
    print(new_df.w.value_counts()[:10])



def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    flip, angle = Flipped(image_obj)
    if flip:
        image_obj = image_obj.rotate(angle, expand=True)
        # print("dm")
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    # cropped_image.show()

def read_img(file_name):
    img = Image.open(data_dir + file_name)
    width, height = img.size
    print(file_name, width, height)
    ori_size = (width,height)
    return img, ori_size


run()

# df_cols = ['Dir','x','y','h','z','Label']
# new_df = pd.DataFrame(columns=df_cols)

# df = new_df.append({'Dir': 1, 'x': 2}, ignore_index=True)


# print(df)
