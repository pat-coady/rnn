"""
Helpful class for evaluating vector representation of words.
Written by Patrick Coady (pcoady@alum.mit.edu)

The WordVector class contains several methods for examining word
embeddings:

    1. Given a word, return closest words (cosine similarity or
    euclidean distance).
    2. Return n most common words.
    3. Compute analogies: a is to b; as c is to ?
    4. Use t-SNE to project list of words into 2 dimensions for
    visualization
"""
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import cdist
import pickle


class WordVector(object):
    def __init__(self, embed_matrix, dictionary):
        """
        Initialize WordVector object. Assumes both dictionaries are built
        such that word value 0 is most common word and word
        value=(len(dictionary)-1) is the least common word.
        :param embed_matrix: 2D numpy array. Row i is the embedding vector
        for word i.
        :param dictionary: Keys word strings. Values are the integer mapping
        of word. Case sensitive.
        """
        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}

    def n_closest(self, word, num_closest=5, metric='cosine'):
        """
        Given a word (in string format), return list of closest words based
        on 1 of 4 distance measures. If word does not appear in dictionary,
        returns empty list.
        :param word: Word string that appears in dictionary.
        :param num_closest: Number of closest words to return.
        :param metric: (as passed to scipy.spatial.distance.cdist):
            'euclidean'
            'cosine'
            (see scipy.spatial.distance for more distance options)
        :return: List of num_closest words, rank ordered from most to least
        similar.
        """
        wv = self.get_vector_by_name(word)
        closest_indices = self.closest_row_indices(wv, num_closest + 1, metric)
        word_list = []
        for i in closest_indices:
            word_list.append(self._reverse_dictionary[i])
        if word in word_list:
            word_list.remove(word)  # remove search word from closest list

        return word_list

    def words_in_range(self, start, end):
        """
        Returns list of words with dictionary values starting at start and
        ending at (end-1).
        :param start: Start of slice (inclusive)
        :param end: End of slice (exclusive)
        :return: List of words (as strings)
        """
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])

        return word_list

    def most_common(self, num=5):
        """
        Returns n most common words.
        :param num: Number of words to return
        :return: List of words (as strings)
        """
        return self.words_in_range(0, num)

    def analogy(self, a, b, c, num=1, metric='cosine'):
        """
        Predict d in the analogy: a is to b; as c is to d.
        :param a: Word string in dictionary
        :param b: Word string in dictionary
        :param c: Word string in dictionary
        :param num: Number of candidates to return for d.
        :param metric: distance measure to us in predicting word d:
            (see n_closest() for options)
        :return: List of words (as strings)
        """
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])

        return d_word_list

    def project_2d(self, start, end):
        """
        Projects word embeddings into 2 dimensions for visualization (using
        t-SNE algorithm as implemented in sklearn). Projects words in range
        [start, end).
        The sklearn TNSE algorithm is memory intensive. Projecting more than
        ~500 words at a time can cause memory overflow.
        :param start: Start of slice (inclusive)
        :param end: End of slice (exclusive)
        :return: tuple:
            2D numpy array of shape = (start-end, 2)
            List of words, length = (start-end), corresponds to row ordering
                of returned numpy array
        """
        tsne = TSNE(n_components=2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])

        return embed_2d, word_list

    def num_words(self):
        """
        Returns number of words in word embedding
        :return: int
        """
        return len(self._dictionary)

    def get_vector_by_name(self, word):
        """
        Return 1D numpy word vector by word. Word must exist in dictionary
        provided during object construction.
        :param word: String
        :return: NP array
        """
        return np.ravel(self._embed_matrix[self._dictionary[word], :])

    def get_vector_by_num(self, num):
        """
        Return 1D numpy word vector by num (word integer value). Word integer
        value must correspond to row number in embed_matrix providing during
        object construction.
        :param num: int
        :return: NP array
        """
        return np.ravel(self._embed_matrix[num, :])

    def get_dict(self):
        """
        Return copy of word dictionary
        :return: dict()
        """
        return self._dictionary.copy()

    def get_reverse_dict(self):
        """
        Return copy of reverse word dictionary
        :return: dict()
        """
        return self._reverse_dictionary.copy()

    def get_embed(self):
        """
        Return copy of 2D numpy embedding matrix
        :return: 2D np array
        """
        return self._embed_matrix.copy()

    def closest_row_indices(self, wv, num, metric):
        """
        Return num closest row indices in sorted order (from closest
        to furthest.
        :param wv: word vector to measure distance from
        :param num: number of indices to return
        :param metric: (as passed to scipy.spatial.distance.cdist):
            'euclidean'
            'cosine'
            (see scipy.spatial.distance for more distance options)
        :return: np array of integer indices (can be used for indexing
        embed_matrix).
        """
        dist_array = np.ravel(cdist(self._embed_matrix, wv.reshape((1, -1)),
                                    metric=metric))
        sorted_indices = np.argsort(dist_array)

        return sorted_indices[:num]

    def save(self, filename):
        """
        Save learned WordVector to file for future fast load.
        :param filename: Filename (with path, if needed) for save.
        :return: None
        Note: no unit test coverage
        """
        embedding = {'embed_matrix': self._embed_matrix,
                     'dictionary': self._dictionary}
        with open(filename + '.p', 'wb') as f:
            pickle.dump(embedding, f)

    @staticmethod
    def load(filename):
        """
        Load WordVector from file and return WordVector object
        :param filename: Filename as used in WordVector.save() method
        :return: WordVector object
        Note: no unit test coverage
        """
        with open(filename + '.p', 'rb') as f:
            embedding = pickle.load(f)

        return WordVector(embedding['embed_matrix'],
                          embedding['dictionary'])

