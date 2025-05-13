# GloBERTise-eval
Contains parts of nlp-binary-event-detection to evaluate different pre-trained GloBERTise models

## Model mapping
Models are referred to differently in this code than on Huggingface

| Huggingface          | Code               | 
|----------------------|--------------------|
| GloBERTise-v01       | globertise         |   
| GloBERTise-v01-rerun | globertisererun    |  
| GloBERTise           | globertisev02      |  
| GloBERTise-rerun     | globertisev02rerun |

## Evaluation
This repo contains code to parse the predictions made by the four different models on a token level binary event detection task. 
The code that was used to finetune the models is in [this repo](https://github.com/globalise-huygens/nlp-binary-event-detection-models).
This repo contains all predictions that were made with the finetuned models and code to analyse them.
For more info on different datasplits and different seeds, see the readme of the beforementioned repo. 

