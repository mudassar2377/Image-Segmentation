Project Details:
Dataset was given with each image of size 256x256 with their repective masks.
The dataset given was divided into three major classes:
• BCC – Basal Cell Carcinoma
• IEC – Intra-epidermal Carcinoma
• SCC – Squamous Cell Carcinoma
Project had two parts Segmentation and Classification.
**Segmentation**:
There were 12 different regions which were to be segemented.
**Classification :**
Next part was to classify these images into three classes given above.

This repo contains semester project which was to segment retinal images using UNet model. Loss used in this model is categorical crosstropy with mean_IOU as metric. Due to less amount of dataset highest score obtained using mean_IOU was 0.56. 
The colab note book also contains CNN model which is used for classification of 3 different eye dieases. The model and it's training with confussion matrix is already ploted in it.
