# Introduction to natural language processing concepts

In order for computer systems to interpret the subject of a text in a similar way humans do, they use natural language processing (NLP), an area within AI that deals with understanding written or spoken language, and responding in kind. Text analysis describes NLP processes that extract information from unstructured text.

Some common NLP text analysis use cases are:

1. **Speech-to-text and text-to-speech conversion**. For example, generate subtitles for videos.
2. **Machine translation**. For example, translate text from English to Japanese.
3. **Text classification**. For example, label an email as spam or not spam.
4. **Entity extraction**. For example, extract keywords or names from a document.
5. **Question answering**. For example, provide answers to questions like "What is the capital of France?"
6. **Text summarization**. For example, generate a short one-paragraph summary from a multi-page document.

Historically, NLP has been challenging as our language is complex and computers find it hard to understand text. In this module, you learn how developments in AI and specifically NLP have led to the models we use today.

## Understand how language is processed

Some of the earliest techniques used to analyze text with computers involve statistical analysis of a body of text (a corpus) to infer some kind of semantic meaning. Put simply, if you can determine the most commonly used words in a given document, you can often get a good idea of what the document is about.

### Tokenization

The first step in analyzing a corpus is to break it down into tokens. For the sake of simplicity, you can think of each distinct word in the training text as a token, though in reality, tokens can be generated for partial words, or combinations of words and punctuation.

For example, consider this phrase from a famous US presidential speech: *"we choose to go to the moon"*. The phrase can be broken down into the following tokens, with numeric identifiers:

Notice that "to" (token number 3) is used twice in the corpus. The phrase "we choose to go to the moon" can be represented by the tokens {1,2,3,4,3,5,6}.

We've used a simple example in which tokens are identified for each distinct word in the text. However, consider the following concepts that may apply to tokenization depending on the specific kind of NLP problem you're trying to solve:

- **Text normalization	Before generating tokens**, you may choose to normalize the text by removing punctuation and changing all words to lower case. For analysis that relies purely on word frequency, this approach improves overall performance. However, some semantic meaning may be lost - for example, consider the sentence "Mr Banks has worked in many banks.". You may want your analysis to differentiate between the person "Mr Banks" and the "banks" in which he has worked. You may also want to consider "banks." as a separate token to "banks" because the inclusion of a period provides the information that the word comes at the end of a sentence
- **Stop word removal**	Stop words are words that should be excluded from the analysis. For example, "the", "a", or "it" make text easier for people to read but add little semantic meaning. By excluding these words, a text analysis solution may be better able to identify the important words.
- **n-grams**	Multi-term phrases such as "I have" or "he walked". A single word phrase is a unigram, a two-word phrase is a bi-gram, a three-word phrase is a tri-gram, and so on. By considering words as groups, a machine learning model can make better sense of the text.
- **Stemming**	A technique in which algorithms are applied to consolidate words before counting them, so that words with the same root, like "power", "powered", and "powerful", are interpreted as being the same token.

## Understand statistical techniques for NLP

Two important statistical techniques that form the foundation of natural language processing (NLP) include: **Naïve Bayes and Term Frequency - Inverse Document Frequency (TF-IDF)**

### Understanding Naïve Bayes
Naïve Bayes is a statistical technique that was first used for email filtering. To learn the difference between spam and not spam, two documents are compared. Naïve Bayes classifiers identify which tokens are correlated with emails labeled as spam. In other words, the technique finds which group of words only occurs in one type of document and not in the other. The group of words is often referred to as bag-of-words features.

For example, the words miracle cure, lose weight fast, and anti-aging may appear more frequently in spam emails about dubious health products than your regular emails.

Though Naïve Bayes proved to be more effective than simple rule-based models for text classification, it was still relatively rudimentary as only the presence (and not the position) of a word or token was considered.

### Understanding TF-IDF

The Term Frequency - Inverse Document Frequency (TF-IDF) technique had a similar approach in that it compared the frequency of a word in one document with the frequency of the word in a whole corpus of documents. By understanding in which context a word was being used, documents could be classified based on certain topics. TF-IDF is often used for information retrieval, to help understand which relative words or tokens to search for.

Simple frequency analysis in which you simply count the number of occurrences of each token can be an effective way to analyze a single document, but when you need to differentiate across multiple documents within the same corpus, you need a way to determine which tokens are most relevant in each document. TF-IDF calculates scores based on how often a word or term appears in one document compared to its more general frequency across the entire collection of documents. Using this technique, a high degree of relevance is assumed for words that appear frequently in a particular document, but relatively infrequently across a wide range of other documents.

## Understand semantic language models

As the state of the art for NLP has advanced, the ability to train models that encapsulate the semantic relationship between tokens has led to the emergence of powerful deep learning language models. At the heart of these models is the encoding of language tokens as vectors (multi-valued arrays of numbers) known as embeddings.

Vectors represent lines in multidimensional space, describing direction and distance along multiple axes. Overall, the vector describes the direction and distance of the path from origin to end. Semantically similar tokens should result in vectors that have a similar orientation – in other words they point in the same direction. As a simple example, suppose the embeddings for our tokens consist of vectors with three elements, for example:

The embedding vectors for "dog" and "puppy" describe a path along an almost identical direction, which is also fairly similar to the direction for "cat". The embedding vector for "skateboard" however describes journey in a very different direction.

The language models we use in industry are based on these principles but have greater complexity. For example, the vectors used generally have many more dimensions. There are also multiple ways you can calculate appropriate embeddings for a given set of tokens. Different methods result in different predictions from natural language processing models.

A generalized view of most modern natural language processing solutions is shown in the following diagram. A large corpus of raw text is tokenized and used to train language models, which can support many different types of natural language processing task.

### Machine learning for text classification
Another useful text analysis technique is to use a classification algorithm, such as logistic regression, to train a machine learning model that classifies text based on a known set of categorizations. A common application of this technique is to train a model that classifies text as positive or negative in order to perform sentiment analysis or opinion mining.

For example, consider the following restaurant reviews, which are already labeled as 0 (negative) or 1 (positive):

- *The food and service were both great*: 1
- *A really terrible experience*: 0
- *Mmm! tasty food and a fun vibe*: 1
- *Slow service and substandard food*: 0

With enough labeled reviews, you can train a classification model using the tokenized text as features and the sentiment (0 or 1) a label. The model will encapsulate a relationship between tokens and sentiment - for example, reviews with tokens for words like "great", "tasty", or "fun" are more likely to return a sentiment of 1 (positive), while reviews with words like "terrible", "slow", and "substandard" are more likely to return 0 (negative).

