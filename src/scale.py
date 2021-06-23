"""
References:
[1] stackoverflow.com/questions/59308710/iou-of-multiple-boxes
[2] math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
[3] ronny.rest/tutorials/module/localization_001/iou/
"""

import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
from tqdm import tqdm
from utils.paths import path_downloads, path_metadata, path_scale

def scale(df, multi):
    df = df[df['multi'] == multi]
    filepaths = df['filepath'].unique()
    if multi == False:
        scale = list((df['x_max'] - df['x_min']) * (df['y_max'] - df['y_min']))
    else:
        scale = []
        for filepath in tqdm(filepaths):
            df_boxes = df[df['filepath'] == filepath]
            boxes = np.array(df_boxes[['x_min', 'y_min', 'x_max', 'y_max']])
            boxes = [shapely.geometry.box(*b) for b in boxes]
            union = shapely.ops.unary_union(boxes)
            scale.append(union.area)
    return pd.DataFrame(data={'filepath':filepaths, 'scale':scale})

path_bounding_boxes = path_downloads/'imagenet_2012_bounding_boxes.csv'
column_names = ('filepath', 'x_min', 'y_min', 'x_max', 'y_max')
df = pd.read_csv(path_bounding_boxes, names=column_names, header=None)

df['multi'] = df.duplicated(subset=['filepath'], keep=False)
df_single = scale(df, multi=False)
df_multi = scale(df, multi=True)

df_scale = pd.concat((df_single, df_multi), ignore_index=True)
df_scale['wnid'] = df_scale['filepath'].str.split('_', expand=True)[0]
df_scale['filepath'] = df_scale['wnid'] + '/' + df_scale['filepath']
df_scale.sort_values('filepath', inplace=True, ignore_index=True)
path_scale_image = path_metadata/'imagenet_image_scale.csv'
df_scale[['filepath', 'scale']].to_csv(path_scale_image, index=False)

df_scale_median = df_scale.groupby('wnid').median()
df_scale_median.reset_index(inplace=True)
np.savetxt(path_scale, list(df_scale_median['scale']), fmt='%.18f')
