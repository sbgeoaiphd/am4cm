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
