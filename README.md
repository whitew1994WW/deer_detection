# deer_detection
For training YOLO-tiny on detecting two classes:

- Deer
- Non-deer animal

## Dataset
The data set used is available on HuggingFace - "myyyyw/NTLNP", and can be downloaded as shown in the notebook.

The dataset includes photos of deer and other animals at night and during the day. 

The dataset annotations are in the Yolo format, but to make it easier to work with, I have converted it to the COCO format and created seperate annotation files for the train/test/val splits.

## Training
The model is trained using the Hugging Face API and the pytorch lightning module. See the train_deer notebook.

The notebook is based on this notebook:

https://colab.research.google.com/drive/1kfU7TpWYrZKRR8eGwdvKgpoEPUtC-J6I?ref=blog.roboflow.com#scrollTo=S2Pf11EvhWT5

## Evaluation

The model is evaluated using IoU, and achieves a 0.76 IoU on the test set. Given the small model size, this is not too bad for a proof of concept.

## Video Annotation

The video annotation is dont in the annotate_video.ipynb notebook. It will annotate a video with the bounding boxes at a given frequency, then save the output videos for review. 
