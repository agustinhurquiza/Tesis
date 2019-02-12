# Segmentation as selective search

## Abstract
For object recognition, the current state-of-the-art is based on exhaustive search. However, to enable the use of more expensive features and classifiers and thereby progress beyond the state-of-the-art, a selective search strategy is needed. Therefore, we adapt segmentation as a selective search by reconsidering segmentation: We propose to generate many approximate locations over few and precise object delineations because (1) an object whose location is never generated can not be recognised and (2) appearance and immediate nearby context are most effective for object recognition. 
Our method is class-independent and is shown to cover 96.7% of all objects in the Pascal VOC 2007 test set using only 1,536 locations per image. Our selective search enables the use of the more expensive bag-of-words method which we use to substantially improve the state-of-the-art by up to 8.5% for 8 out of 20 classes on the Pascal VOC 2010 detection challenge.

## Info
This is the Segmentation as selective search objectness proposal estimator implementation,We would appreciate if you could cite and refer to  the papers below.

```
@InProceedings{vandeSandeICCV2011,
  author       = "van de Sande, K. E. A. and Uijlings, J. R. R. and Gevers, T. and Smeulders, A. W. M.",
  title        = "Segmentation As Selective Search for Object Recognition",
  booktitle    = "IEEE International Conference on Computer Vision",
  year         = "2011",
  url          = "https://ivi.fnwi.uva.nl/isis/publications/2011/vandeSandeICCV2011",
  pdf          = "https://ivi.fnwi.uva.nl/isis/publications/2011/vandeSandeICCV2011/vandeSandeICCV2011.pdf",
  has_image    = 1
}
```
github: https://github.com/ranchirino/segmentation-as-selective-search
