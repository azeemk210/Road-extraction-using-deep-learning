# Road-extraction-using-U-Net
U-Net algorithm to extract road mask from satellite images

This project implements a deep learning model using U-Net architecture for road extraction from satellite images. The goal is to accurately identify the roads in a satellite image, which can be useful for various applications such as urban planning, disaster response, and traffic management.

Data Preparation
The data preparation process includes the following steps:

Image pre-processing: This step involves resizing the images to a standard size, converting them to grayscale, and normalizing the pixel values to improve model performance.

Image augmentation: This step involves applying various techniques such as rotation, flipping, and zoom to artificially increase the size of the training dataset and improve model generalization.

Model Implementation
The deep learning model is implemented using the U-Net architecture, which is widely used for image segmentation tasks. The U-Net consists of a contracting path and an expanding path, where the contracting path captures the context of the image, and the expanding path restores the spatial information lost in the contracting path.

Training and Evaluation
The model is trained on a large dataset and evaluated using various metrics, including accuracy, precision, recall, and F1 score. The evaluation results show that the model performs well in identifying the roads in satellite images.
