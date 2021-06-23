import os
import pathlib

path_conv5_hdd = pathlib.Path(os.environ['PATH_CONV5_HDD'])
path_conv5_ssd = pathlib.Path(os.environ['PATH_CONV5_SSD'])
path_downloads = pathlib.Path(os.environ['PATH_DOWNLOADS'])
path_expt = pathlib.Path(os.environ['PATH_EXPERIMENT'])
path_expt_raw = pathlib.Path(os.environ['PATH_EXPERIMENT_RAW'])
path_figures = pathlib.Path(os.environ['PATH_FIGURES'])
path_images = pathlib.Path(os.environ['PATH_IMAGES'])
path_metadata = pathlib.Path(os.environ['PATH_METADATA'])
path_objdet = pathlib.Path(os.environ['PATH_OBJDET'])

path_clutter = path_metadata/'imagenet_class_clutter.txt'
path_confusion = path_metadata/'imagenet_class_confusion.txt'
path_cooccurrence = path_metadata/'imagenet_class_cooccurrence_openimages.txt'
path_difficulty = path_metadata/'imagenet_class_difficulty.txt'
path_mids = path_metadata/ 'openimages_class_mids.txt'
path_names = path_metadata/'imagenet_class_names.txt'
path_scale = path_metadata/'imagenet_class_scale.txt'
path_similarity_cnn = path_metadata/'imagenet_class_similarity_cnn.txt'
path_similarity_human = path_metadata/'imagenet_class_similarity_human.txt'
path_similarity_semantic = path_metadata/'imagenet_class_similarity_semantic.txt'
path_wnids = path_metadata/'imagenet_class_wnids.txt'
