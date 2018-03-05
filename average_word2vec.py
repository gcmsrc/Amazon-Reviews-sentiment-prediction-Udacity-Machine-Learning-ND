"""

	This module contains the functions to transform
	a review into the average of its word vectors
	
"""

import numpy as np

def vectorise_review(review, model, vectors_vocab, wv_dimension):
    
    """
    
        Add word vectors to a zero-vector review. It then averages the vector out
        by dividing by the number of added word vectors.
        
        Args:
            - review: a string
            - model: a gensim word2vec model
            - vectors_vocab: set
            - wv_dimension: int, the dimension of word vectors in the word2vec model
            
        Returns:
            - review_vector: numpy array
    
    """
    
    # Initialise a zero vector for the review
    review_vector = np.zeros((wv_dimension,), dtype='float32')
    
    # Initialise counter
    counter = 0
    
    # Loop to update review_vector with word vectors
    for w in review.split():
        if w in vectors_vocab:
            review_vector += model.wv[w]
            counter += 1

    # Average out
    review_vector = np.divide(review_vector, counter)
    
    return review_vector

def vectorise_review_set(reviews, model, vectors_vocab, wv_dimension):
    
    """
    
        Transform a set of reviews into their corresponding averages of word
        vectors.
        
        Args:
            - reviews: a list of strings
            - model: a gensim word2vec model
            - vectors_vocab: set
            - wv_dimension: int, the dimension of word vectors in the word2vec model
    
    """
    
    # Initialise set of review vectors
    reviews_vectors = np.zeros((len(reviews), wv_dimension), dtype='float32')
    
    # Loop through the reviews
    for i, r in enumerate(reviews):
        
        reviews_vectors[i] = vectorise_review(r, model, vectors_vocab, wv_dimension)
        
    return reviews_vectors
