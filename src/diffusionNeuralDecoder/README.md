Using the LibreSpeech Dataset


## Dataset info and Choices
### Competition Dataset:
#### Brain Sequence Bins:
- Max = 919
- Median = 295
- Mean = ~311
- 99th Percentile = 715
Means the Median time per trial is 295*20/1000 = 5.9 seconds

#### Phoneme Bins
- Max = 77
- Median = 24
- Mean = 24.8
- 99th Percentile = 62

#### Character/Words
- Max = 87 Characters
- Median = 29 Characters
- Mean = 29 Characters
- 99th Percentile = 72

### Pretraining Dataset Options:
The choice was to use the LibriSpeech ASR Dataset, the choice was so that the phoneme lenght was on average longer than the BCI Phoneme Pairs. This is because the Pretraining dataset will be used to generate the unconditional model and the BCI Phonemes will be primarily used as finetuning. In a fixed window diffusion model, this would mean that the maximum BCI Phoneme length of the data used should be at most the maximum Phoneme length of the pretraining datapoints.
The LibriSpeech ASR dataset was chosen as it was semantically similar to the BCI dataset and designed for spoken word. General text corpa could've been used such as OpenWebText but there might've been a confounder on the semantic properties of the text data as spoken versus written vernacular and structure are very different. 

We are using the train.clean.360 split from the LibriSpeech set which is 104k rows long.
#### LibriSpeech ASR Character length
Median is between 180 to 219 words

Good Range + greater by a large margin of the character lengths. Will likely use max 200 character length from this dataset. This isn't the most well specified as there aren't as many sentences in the same length/domain as the BCI dataset, but we will have to make due

For the limits of this project, 200 characters was chosen for compute time using around 65k rows, however, later tests can be made using larger datasets numbers. 
