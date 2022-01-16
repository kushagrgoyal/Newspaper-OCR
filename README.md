# Newspaper-OCR

This repository is to extract text from very old newspapers, such as from 1800s or 1900s. This code has combination of Computer Vision techniques and NLP techniques to first, extract text data in columns and then arrange them in correct order, mostly.

It is especially tricky extracting text from these newspapers because of the following reasons:
- The newspapers are scanned images so OCR will have to used to extract text
- The text is arranged in columns so any direct use of any OCR tool is pretty useless
- The text is also arranged in an awkward way in these old newspapers thereby making the task of identifying correct articles quite tricky

## Model Structure
### Computer Vision section
The backbone of the project is the YOLOv5 model taken from https://github.com/ultralytics/yolov5

An Object Detection model was trained on the custom dataset to be able to identify the text columns in a newspaper. The training images were taken from Newspaper Navigator dataset here https://news-navigator.labs.loc.gov/

These images manually labelled to only extract news articles. No ads, cartoons or images were considered for training the model as the object was to extract the news articles.
### NLP Section
The second stage of the model is to use a Sentence transformer model to be able to identify which text columns actually are part of the same article. The model is taken from https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2

The articles in these old newspapers are not arranged as nicely as they are in modern newspapers. Here, the articles can be arranged in quite non-intuitive manner so much so that manu a times even a human reading could get confused as to which text section is the next part of the same article.

The Sentence Transformers were quite helpful to figure that out by identifying nearest "Text-boxes" and then grouping them into articles. Still does not work a 100 % of the time.
### Final Output
The output of the Computer vision model looks like the following image:
![Final Image](https://github.com/kushagrgoyal/Newspaper-OCR/blob/main/script_outputs/rough12_small.png "Final Image")

The output from the script are the images such as the one attached above and then the actual .txt file containing the text in the order shown in the image.
