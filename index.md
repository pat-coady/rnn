---
layout: default
comments: true
---

# Overview

My primary objective with this project was to learn [TensorFlow](https://www.tensorflow.org/). I've previously used [Keras](https://keras.io/) with TensorFlow as its back-end. Recently, Keras couldn't easily build the neural net architecture I wanted to try. So it was time to learn the TensorFlow API.

I chose to build a simple word-embedding neural net. This seemed a good compromise that was interesting, but not too complex. I didn't want to simultaneously debug my neural net and my TensorFlow code.

# Word Vectors


# Unsupervised Learning


# Learning Word Vectors


# Results


#### Word Similarity

# The Code

The project has 2 major components:

* 3 Python modules to:
	1. Load and process text documents (docload.py)
	2. Build and train TensorFlow model (windowmodel.py)
	3. Explore the learned word vectors (wordvector.py)
* iPython Notebooks
	1. Load Sherlock Holmes books, train models and explore the results (sherlock.ipynb)
	2. Hyper-parameter tuning and viewing learning curves (tune\_\*.ipynb)
	3. Plot word frequencies (word\_frequency.ipynb)

I tried to write clear code for the 3 Python modules. I hope you find them useful for your own work, or just to better understand how to use TensorFlow.

For additional details on the code please see [README.md](https://github.com/pat-coady/word2vec/blob/master/README.md) on the GitHub page.

{% if page.comments %}
<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
this.page.url = 'https://pat-coady.github.io/rnn/';
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = '//https-pat-coady-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}
