# Understanding top-down attention using task-oriented ablation design

![](/figures/task_oriented_ablation_design.png)

## Abstract
Top-down attention allows neural networks, both artificial and biological, to focus on the information most relevant for a given task. This is known to enhance performance in visual perception. But it remains unclear how attention brings about its perceptual boost, especially when it comes to naturalistic settings like recognising an object in an everyday scene. What aspects of a visual task does attention help to deal with? We aim to answer this with a computational experiment based on a general framework called task-oriented ablation design. First we define a broad range of visual tasks and identify six factors that underlie task variability. Then on each task we compare the performance of two neural networks, one with top-down attention and one without. These comparisons reveal the task-dependence of attention’s perceptual boost, giving a clearer idea of the role attention plays. Whereas many existing cognitive accounts link attention to stimulus-level variables, such as visual clutter and object scale, we find greater explanatory power in system-level variables that capture the interaction between the model, the distribution of training data and the task format. This finding suggests a shift in how attention is studied could be fruitful. We make publicly available our code and results, along with statistics relevant to ImageNet-based experiments beyond this one. Our contribution serves to support the development of more human-like vision models and the design of more informative machine-learning experiments.

## Reproduce this project from scratch
The commands below generate the contents of `data/` and `figures/` from scratch using the code in `src/`. Running them requires a [Conda](https://docs.conda.io) installation and a local copy of [ImageNet](http://www.image-net.org). We provide estimated completion times for all Python scripts, assuming a setup with one GPU and with precomputed image representations saved on a fast storage drive.

```shell
# SETUP ------------------------------------------------------------------------
# Set the number of tasks to include in the experiment.
N_TASKS=2000

# Tell CUDA which GPU to use.
read -p 'Which GPU? ' GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU

# Tell TensorFlow to only print warnings and errors, not info messages.
export TF_CPP_MIN_LOG_LEVEL=1

# Make a local copy of the repository and use it as the working directory.
PATH_REPO=~/attention-v2/
git clone https://github.com/fbickfordsmith/attention-v2.git $PATH_REPO
cd $PATH_REPO

# Build and activate a Conda environment.
conda env create -f environment.yml
source activate attention_toad

# Set up a directory for the metadata used in the experiment.
export PATH_METADATA=$PATH_REPO/data/metadata/
mkdir -p $PATH_METADATA

# Get some of the metadata in raw form from online sources.
export PATH_DOWNLOADS=$PATH_REPO/data/downloads/
mkdir -p $PATH_DOWNLOADS
URL1=https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
URL2=https://repository.prace-ri.eu/git/Data-Analytics/Benchmarks/-/raw/bfea8c2c69078a98c35a6e774809e7d9cc807874/ImageNetUseCaseV2/Dataset/Metadata/imagenet_2012_bounding_boxes.csv
URL3=https://raw.githubusercontent.com/cvjena/semantic-embeddings/master/embeddings/imagenet_mintree.unitsphere.pickle
URL4=https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv
URL5=https://drive.google.com/uc?id=1mKRYfmw5qxLgriWGqS1TJ5M1T1vgjNtU
curl $URL1 -o $PATH_DOWNLOADS/imagenet_class_index.json
curl $URL2 -o $PATH_DOWNLOADS/imagenet_2012_bounding_boxes.csv
curl $URL3 -o $PATH_DOWNLOADS/imagenet_mintree.unitsphere.pickle
curl $URL4 -o $PATH_DOWNLOADS/class-descriptions-boxable.csv
curl $URL5 -L -o $PATH_DOWNLOADS/marg_smat_4-221-6.txt

# Set up a directory for object-detection predictions on ImageNet.
export PATH_OBJDET=$PATH_REPO/data/object_detection/
mkdir -p $PATH_OBJDET

# Set up two directories for experimental data. One is for storing this data in
# its raw form: three files---training metrics, trained parameters, evaluation
# accuracies---for each task in the experiment. The other is for storing the
# same data but in a tidier combined form.
export PATH_EXPERIMENT=$PATH_REPO/data/experiment/
export PATH_EXPERIMENT_RAW=$PATH_REPO/data/experiment_raw/
mkdir -p $PATH_EXPERIMENT $PATH_EXPERIMENT_RAW

# Set up a directory for figures. Download project illustrations.
export PATH_FIGURES=$PATH_REPO/figures/
mkdir -p $PATH_FIGURES
URL6=https://drive.google.com/uc?id=1JnXtfoZGLr0TEBlHgttl9NTdJOqVsqy4
curl $URL6 -L -o $PATH_FIGURES/illustrations.zip
unzip -j $PATH_FIGURES/illustrations.zip -d $PATH_FIGURES
rm $PATH_FIGURES/illustrations.zip

# Point to the ImageNet data. `PATH_IMAGES` contains three directories: `train`,
# `val` and `val_white`. Each of those directories contains 1000 directories,
# one for each class: `n01440764`, `n01443537`, ..., `n15075141`.
export PATH_IMAGES=/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/

# To speed up data-loading and passes through the model, precompute image
# representations at VGG16's final convolutional layer and save as TFRecords.
export INPUT_MODE=conv5

# Due to permission issues, first save image representations to HDD, then
# transfer to SSD. Set up a directory on each drive.
export PATH_CONV5_HDD=$PATH_REPO/data/conv5_tf/
export PATH_CONV5_SSD=/mnt/fast-data16/datasets/freddie/conv5_tf/
mkdir -p $PATH_CONV5_HDD $PATH_CONV5_SSD

# Track progress by printing commands when they are executed.
cd src
set -x

# PREPROCESSING ----------------------------------------------------------------
# Process the ImageNet and OpenImages class names and IDs [<1min].
python imagenet_class_ids.py
python openimages_class_ids.py

# Precompute representations for all images used in the experiment [2hr].
if [[ $INPUT_MODE == conv5 ]]; then
    python representations.py --data_partition train
    python representations.py --data_partition val_white
    cp -r $PATH_CONV5_HDD/* $PATH_CONV5_SSD
fi

# BASELINE NETWORK -------------------------------------------------------------
# Check the model and the data pipeline [5min].
python sanity_check.py

# Define the baseline task: classifying images from all ImageNet classes [<1min].
python tasks.py --n_tasks 0

# Train and test the baseline model [40min].
python train_test.py --task_id 0 --init ones --intensity 1 --eval_gap 5
python tidy.py

# IMAGENET STATISTICS ----------------------------------------------------------
# Run object detection on the training set [200hr].
for (( i=0; i<=999; i+=1 )); do
    python object_detection.py --class_id $i
done

# Using the results of `object_detection.py`, compute a matrix that summarises
# cooccurrences between ImageNet and OpenImages classes [10min].
python cooccurrence.py

# Using the results of `object_detection.py` and `cooccurrence.py`, compute
# image and class clutter scores [40min].
python clutter.py

# Compute class difficulty scores [<1min].
python difficulty.py

# Using downloaded bounding boxes, compute image and class scale scores [5min].
python scale.py

# Compute the baseline model's confusion matrix on the validation set [<1min].
python confusion.py

# Using results from `confusion.py`, compute the CNN-based similarity of each
# pair of classes [<1min].
python similarity_cnn.py

# Process downloaded human similarity judgements [<1min].
python similarity_human.py

# Using downloaded semantic embeddings, compute the semantic similarity of each
# pair of classes [<1min].
python similarity_semantic.py

# Assemble a table summarising the ImageNet statistics [<1min].
python imagenet_class_statistics.py

# EXPERIMENT -------------------------------------------------------------------
# Define a collection of tasks [5min].
python tasks.py --n_tasks $N_TASKS

# Train and test models on the tasks. In order to speed up training, use a
# larger `eval_gap` value than for the baseline model. Since the training runs
# are longer here, we can still get satisfactory training curves even if we
# increase `eval_gap` [(0.5*N_TASKS)hr].
for (( TASK_ID=1; TASK_ID<=$N_TASKS; TASK_ID+=1 )); do
    python train_test.py --task_id $TASK_ID --init task0000 --intensity 0.5 --eval_gap 20
done
python tidy.py

# Plot and analyse data summarising the experiment [<1min].
python plot_analyse.py
```

## Cite this work
Please cite [our paper](https://arxiv.org/abs/2106.11339) if you use our code, data or ideas in your work.
```
@article{bickfordsmith21attention,
    author = {Freddie Bickford Smith and Brett D Roads and Xiaoliang Luo and Bradley C Love},
    title = {Understanding top-down attention using task-oriented ablation design},
    journal = {arXiv:2106.11339},
    year = {2021},
}
```

## Get in touch
Contact [Freddie](https://fbickfordsmith.com) if you have any questions about this research or find an error in this repository.
