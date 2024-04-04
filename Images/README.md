# Images

This folder contains all the images that are used in the paper. 

Note the images follow the naming convention of: dataset_resolution_[original/rgb/gray]_samplenumber/classnumber.jpeg
* **dataset** is either: breastmnist, dermamnist, pathmnist
* **resolution** is one of the following values: 28, 64, 128, 224 to denote the resolutions of MedMNIST+
* **[original/rgb/gray]**
  * original denotes that this image has had no editing.
  * rgb denotes that this image has had no editing but has the channels rgb.
  * gray denotes that the image has been transformed from rgb to grayscale
* **samplenumber** is the number of the sample in the MedMNIST+ collection.
* **classnumber** is the label that the sample image represents.
  * *BreastMNIST*:
    * Class0: malignant
    * Class1: normal, benign
     
   * *DermaMNIST*:
     * Class0: actinic keratoses and intraepithelical carcinoma
     * Class1: basal cell carcinoma
     * Class2: benign keratosis-like lesions
     * Class3: dermatofibroma
     * Class4: melaonma
     * Class5: melanocytic nevi
     * Class6: vascular lesons
     
   * *PathMNIST*:
     * Class0: adipose
     * Class1: background
     * Class2: debris
     * Class3: lymphocytes
     * Class4: mucus
     * Class5: smooth muscle
     * Class6: normal colon mucosa
     * Class7: cancer-associated stroma
     * Class8: colorectal adenocarcinoma epithelium

Please feel free to take a look.
