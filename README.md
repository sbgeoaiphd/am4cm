# am4cm
Attention Masks 4 Cloud Masking
Investigating attention masks learned from other tasks repurposed for cloud masking

## Background
Cloud masking is a common pre-processing step when working with satellite imagery including Satellite Image Time Series (SITS) data. It is often considered an essential step when preparing data for use with machine learning models. The goal of cloud masking is to identify and mask out cloudy pixels in the data. Many machine learning architectures are highly sensitive to noisy data (e.g. clouds) and can perform poorly if cloudy data is not properly removed.

Significant research effort is dedicated to developing labelled datasets and cloud masking algorithms (e.g. https://cloudsen12.github.io/).

However, modern deep learning architectures such as transformers have been shown to perform well on downstream SITS tasks without first removing clouds. Marc Rußwurm and Marco Körner  (https://arxiv.org/abs/1910.10536) explored this topic in the context of crop type mapping and showed that both transformers and recurrent neural networks were able to learn to ignore clouds when trained on data with clouds present. This approach of using non-cloud masked data with transformer architectures has been deployed at scale by Regrow Ag (work presented at EO4AGRI 2024 - no public link available).

## Objective
The fact that transformers can learn to ignore clouds via learned attention weights presents an interesting opportunity beyond the ability to simply skip cloud masking in downstream applications. The learned attention masks themselves may be able to be repurposed as cloud predictions and used for cloud masking. This would be interesting as it would allow for learned cloud predictions without the need for labelled data.

This project aims to investigate the feasibility of using attention masks learned from other tasks as cloud masks. The project will involve training a transformer model on a task unrelated to cloud masking (crop type classification) and then evaluating the attention masks learned by the model to see if they can be used as cloud masks.

## Model
This project will use a the "LTAE" model by Garnot (https://arxiv.org/abs/2007.00586) adapted to function on pixel time series (as in Marc Rußwurm and Marco Körner's work mentioned above). I.e. the model takes a 2d array of observations (time) x bands and produces a single classification per pixel.

## Dataset and Task
The model will be trained on the PASTIS (https://arxiv.org/abs/2107.07933 and https://github.com/VSainteuf/pastis-benchmark) Crop Segementation dataset. This dataset contains Sentinel-2 imagery and crop type labels for a region in France. The model will be trained to predict crop types from the imte timeseries. Note: the benchmark was built for development of a spatio-temporal model (U-TAE) but we'll just be using a temporal pixel-wise model.

## Other Objectives
This project will also serve as practice and demonstration of my skills in:
* automated testing (pytest)
* environment management (poetry)
* version control (git)
* pytorch:
    * data loading
    * data transformations
    * model training
    * model inference

## Not included
To limit the scope of work and focus on the key objectives, this project will not include:
* hyperparameter tuning
* experiment tracking
* training tracking


# Environment Setup
## Using real data for testing
To run the tests using real PASTIS .npy files and metadata, you'll need to set an environment variable that points to the folder containing the real data.

1. Set the PASTIS_PATH environment variable:
This variable should point to the base PASTIS directory where the metadata.geojson and DATA_S2 directory are located. For example:
`export PASTIS_PATH="/mnt/c/data/PASTIS-R/PASTIS-R"`

2. Run the tests: Once the environment variable is set, you can run the tests with pytest:
`pytest --cov=src tests/`


# Results
## Crop Classification
The main focus of this project is to investigate the attention masks learned by the model. However, as the supervised task is crop type classification, lets see how the model did.

Note, the model was trained on only one fold combination of PASTIS (1-3), did not undergo hyper parameter tuning or any iteration on the training process to improve performance. These results represent the "first attempt" without further tuning. However, the model did converge during training.

### Results
#### Sample crop classification
![Sample crop classification](/evals/sample_classification.png)]

#### Key metrics 
(See evals/crop_classification_metrics.csv for full results)

Overall accuracy: 70%

Major crop F1 scores:
* Soft Winter Wheat: 81%
* Corn: 86%
* Winter Barley: 71%
* Winter Rapeseed: 81%
* Meadow: 63%

#### Confusion Matrix
![Confusion Matrix](/evals/TEST_best_model_ltae_confusion_matrix_true.png)

### Conclusion
These results are good baseline performance for crop classification in this kind of situation, though they are not cutting edge. The model could likely be improved with hyperparameter tuning. Furthermore, if looking at pixel-wise accuracy for semantic segmentation, spatial operations (e.g. U-TAE) are necessary to reach SotA performance. Alternatively, in some settings field level results are used based on majority pixel classification scores per field. Given that such post-processing removes field boundary misclassifications and in-field noisy predictions, the resulting field-level performance is often significantly higher.

## Attention Masks as Cloud Masks
See the notebook 'notebooks/ltae_inference.ipynb' for full results and conclusions.

Example:
![Attention Masks](/evals/example_att_map.png)

### Main Conclusion:
The model shows the ability, via attention weights, to focus on (green vegetation, bare soil) or ignore (clouds, cloud shadows) particular pixel types at particular points in time. However, the behaviour is complex and time-dependent and doesn't lend itself to a simple thresholding approach for cloud masking. Given the model's ability to attend to particular pixel types, and its ability to mostly ignore cloudy pixels via the attention weights, it is likely this kind of modelling approach (attention based architecture, time series data) would be highly adept at cloud masking if provided with supervised labels for the task. Furthermore, because of the distinct spatial patterns of clouds, any high performance cloud detection model to be used in general applications should consider using spatial operations (in addition to temporal operations) to detect clouds.
