# AutomatedTreeCensus
Repo to that predicts the locations of trees from a satellite tif image. This work was for a consulting project for the city of Athens, Greece, which I worked on as an Insight AI Fellow

## Motivation for this project format:
- **src** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

# Setup
Clone the repository
```
git clone https://github.com/mstefferson/AutomatedTreeCensus
```

## Building info
In order to interact with satellite data (both GIS and tif images), we need to use GDAL (Geospatial Data Abstraction Library). I found the installation of this to be a bit of a pain, so I build a docker image. With the exception of training on a AWS GPU node, all code should be ran through the docker image. More details on training vs processing below.

This docker image is built from thinkwhere/gdal-python. It contains GDAL and many geospatial python libraries. It also supports jupyter notebooks. 

## Docker
With the exception of GPU training, building the docker image, and starting up a Docker container, all code should be ran within a Docker container. The docker images mounts the WD into the container.

### Get docker image
Pull the docker image from my docker hub

```
docker pull mstefferson/tree_bot:latest
```
Or build it using the docker file (done through a bash script)

``` bash
./build_docker
```
### Run docker image
To run a bash shell

```
./docker_run
```
To run a jupyter notebook

``` bash
./build_docker_jn
```

Note, the docker container is removed when exiting

# Data
The raw data I was given was a single tif file with four channel's (R, G, B, IR) of all of Athens. In order to train, I did pretrained on Kaggle's DSTL
[Kaggle's DSTL challenge](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection). Next, we use transfer learning on a subset of self-labeled data. To label data, I used [labelImg](https://github.com/tzutalin/labelImg). 

It should be straightward to use this repo on new data. If you are using this on your own satellite images, I recommend pretraining with DSTL (insructions below) and use transfer learning on (hopefully pre-) labeled data.

## Prepping DSTL data

This repo can take the DSTL, and process it to a useable format for training. It breaks up the giant tif files into smaller pngs and grabs the labels out of the geojson files and puts them into a usable format. Currently, the model uses VOC label format, but the code can produce YOLO label format as well. 

- Download the DSTL [data](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)
- Unzip and place all of the contents in the data path of your liking. I recommend data/raw/dstl
- Feel free to delete the 16 band data, I don't use it
- Edit config file, configs/config_dstl.json so that it includes the correct paths
- Start a Docker container
- Run the code to clean the data
```
python src/preprocess/clean_dstl/build_dstl_dataset.py -c configs/config_dstl.json
```

## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint 
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
jk
