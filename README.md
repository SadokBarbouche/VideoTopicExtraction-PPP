# TED Talks Analysis

This repository contains code for analyzing TED Talks transcripts using Natural Language Processing techniques. It includes preprocessing the data, applying topic modeling with Latent Dirichlet Allocation (LDA), and labeling topics using Word2Vec. The goal is to extract insights and discover meaningful topics from the TED Talks dataset.

## <a href="https://www.kaggle.com/datasets/rounakbanik/ted-talks?select=transcripts.csv"> Dataset</a>

The dataset used for this analysis is the TED Talks dataset, which includes transcripts and URL information about all audio-video recordings of TED Talks uploaded to the official TED.com website until September 21st, 2017. The dataset is a valuable resource for natural language processing and machine learning research, providing a large amount of text data along with additional metadata about the talks.

## Data Preprocessing

The preprocessing steps include tokenization, stopwords removal, and lemmatization. The text data is processed using the spaCy library to tokenize the text, remove stopwords, and lemmatize the tokens. The preprocessed text is then further processed using the Gensim library's simple_preprocess function.

## Creating Dictionary and Corpus

The preprocessed texts are transformed into bigrams and trigrams using the Gensim library's Phrases model. The resulting bigram and trigram models are then applied to the preprocessed texts. A dictionary is created from the preprocessed texts, and a corpus (bag of words representation) is generated using the dictionary.

TF-IDF (Term Frequency-Inverse Document Frequency) is calculated for the corpus to identify the most important words in each document. The low-value words and words missing in TF-IDF are removed from the corpus.

## LDA Model

The LDA (Latent Dirichlet Allocation) model is trained with the optimal number of topics determined using the coherence score. The number of topics is set to 17. The LDA model is trained on the corpus using the Gensim library's LdaModel.

## Visualizing the Topics

The topics extracted by the LDA model are visualized using the pyLDAvis library. The visualization provides an interactive display of the topics, their coherence scores, and their relationships.

## Word2Vec Model

A Word2Vec model is trained on the bigram and trigram data to capture word embeddings. The Word2Vec model is used to find similar words based on a given keyword.

## Topic Labeling using Word2Vec

The topics extracted by the LDA model are labeled using Word2Vec. Similar words are identified for each topic keyword, and the labeled topics are generated.

## Testing the Model

To test the model, you can provide a TED Talk transcript from the dataset. The model will predict the topic to which the transcript belongs based on the relevance score. The predicted topic and its corresponding keywords will be displayed.

## Requirements

To run the code in this repository, you need the following libraries:

- NumPy
- Gensim
- SpaCy
- NLTK
- pyLDAvis
- pandas
- WordCloud
- Matplotlib

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/SadokBarbouche/VideoTopicExtraction-PPP
2. For a better user experience , you may want to install jupyter notebook (VSCode can nevertheless do the job)      

## To learn more about our project , check the folloiwing link : https://drive.google.com/drive/folders/1SHHJRXg2_L4ahrHNclFr84PjW7wwU07h


