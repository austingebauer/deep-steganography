# deep-steganography

Project for CSE 576, Computer Vision, at the University of Washington.

The goal of the Deep steganography project is to implement the research paper,
[Hiding Images in Plain Sight: Deep Steganography](https://papers.nips.cc/paper/2017/file/838e8afb1ca34354ac209f53d90c3a43-Paper.pdf),
which was introduced at NIPS in 2017. The paper outlines a method to visually hide, and
eventually reveal, a full N × N × RGB pixel secret image in another N × N × RGB cover image,
with minimal distortion to the cover image. To do so, deep neural networks will be used for
hiding and revealing the secret image. The neural networks will be trained on images from
the [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) database.

## Setup

Unzip dataset:
```
unzip tiny-imagenet-200.zip
```

Create Conda Environment:
```
conda create --name deep-steganography --file requirements.txt
conda activate deep-steganography
```

## Usage

Run python program to train the neural network:
```
python main.py
```
