---
layout: default
comments: true
---

# Objectives

1. Build practical experience designing and training RNNs
2. Compare quality of word vectors learned by RNN vs. [Continuous Bag-of-Words (CBOW)](https://arxiv.org/pdf/1301.3781.pdf)
3. Learn TensorBoard

I went into some detail on Word Vectors and Unsupervised Learning in my [Word2Vec with TensorFlow Project](https://pat-coady.github.io/word2vec/) - so I won't repeat that here. 

# Architecture

1. An embedding layer between one-hot word encoding and RNN
2. RNN: Basic RNN (no memory), GRU or LSTM cell
3. The last output of the RNN is picked off and fed to a hidden layer
4. N-way softmax (N = vocabulary size)

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![RNN Diagram](assets/rnn_diagram.png)
</div>

With this skeleton in place, there are still many decisions to explore:

    * RNN cell: BasicRNN (no memory), LSTM or GRU
    * Layer sizes: embedding layer, number RNN cells, hidden layer
    * RNN length: number of steps to unroll RNN

One decision I made at the start was to learn sequences of words versus sequences of characters. This was partly so I could compare the quality of word vectors from RNNs to CBOW. Also, I wanted to use TensorBoard's embedding visualization. And, finally, I wanted to see how [Candidate Sampling](https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling) loss would perform with a RNN.

# Training

Like my Word2Vec project, I again used 3 Sherlock Holmes books as the model input (courtesy of [Project Gutenberg](https://www.gutenberg.org/)). This choice was convenient because I had already written helpers to load and process this document format.

Training introduces further hyperparameters. To name a few:

    * Optimizer: SGD w/ momentum, Adam and RMSProp
    * Learning rate (and other optimizer controls)
    * Gradient norm clipping

As with the model hyperparameters, I will explore various settings and report the results. As you will see, TensorBoard helps immensely as you explore model settings.

# Results

## Generated Text

Everyone enjoys reading RNN-generated text:

    true as gospel," said he, smiling," but if you are the use of your
    own eyes that you have read me to your hotel, and you have no use
    in the way."" it was in a time."" you have no doubt that you have
    been in the house," said holmes, laughing;" but if you can catch 
    the police, ' said he. 'i have three times in the room. '" 'i had 
    been better, ' said i; 'you have been a right. you must make a note,
    mr. holmes, at the moment when i got home in the morning with my
    father, however, who had shown up the whole death of the man s death.
    
This model used LSTM cells and was trained for 75 epochs. Here are the key settings:

    * embedding layer width = 64
    * rnn width = 192
    * rnn sequence length = 20
    * hidden layer width = 96
    * learning rate = 0.05, momentum = 0.8 (SGD w/ momentum)
    * batch size = 32

## Word Vectors

My evaluation of the learned word vectors is purely qualitative. In that spirit, here are some fun TensorBoard embedding visualizations. This first .gif is an animation of [t-SNE](http://distill.pub/2016/misread-tsne/) running.

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![t-SNE Learning](assets/t-sne-lava-lamp.gif)
</div>

(Incidentally, I have found the PCA does quite poorly on this task. Even with a 3rd dimension, it is rare that 2 closely spaced points are related words.)

I ran t-SNE for 350 iterations (all through the TensorBoard GUI) on the 2,048 most common words. In this animation I search for a cluster of words. Then I select the cluster and check for a relationship:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![t-SNE Learning](assets/explore-embed.gif)
</div>

## RNN vs. CBOW

Here are some synonyms and analogies from the RNN embedding and the CBOW embedding. First synonyms (based on cosine similarity):

**RNN:**

    5 closest words to: 'run'
    ['walking', 'play', 'knowledge', 'happened', 'engaged']
    5 closest words to: 'mr'
    ['st', 'mrs', 'dr', 'continued', 'above']
    
**CBOW:**

    5 closest words to: 'run'
    ['fall', 'sit', 'go', 'break', 'live']
    5 closest words to: 'mr'
    ['mrs', 'dr', 'st', 'boy', 'message']

Now some analogies. First "had" is to "has", as "was" is to ___ :

**RNN:**

    ['was', 'is', 'has', 'am', 'are']
    
**CBOW:**

    ['was', 'is', 'has', 'shows', 'becomes']

I might give a slight edge to RNN. In fairness to CBOW, the RNN was trained 5 times longer. It would be interesting to see how a bidirectional RNN does.

# Hyperparameters

## Layer Sizes

I started with a LSTM cell and some quick exploration to pick a reasonable optimizer and learning rate. Then I checked a grid of layer sizes: embedding layer, rnn layer (width and number of steps) and final hidden layer.

    * embedding layer width = 64
    * rnn width = 192
    * rnn sequence length = 20
    * hidden layer width = 96

At these settings, the model performance was most sensitive to decreasing the hidden layer width. A rnn sequence length of 20 steps is overkill for learning word vectors. Even at a rollout of 5 steps, you learn reasonably good word vectors. But with such a short rollout the model does a terrible job at generating new text - typically repeating the same few words (e.g. 'it is. it is. it is. it is.')

It is subtle, but the lower grouping of training loss curves is with a hidden layer size of 96. The upper grouping of curves have a hidden layer size of 64:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![Loss vs. Architecture Variation](assets/loss_vs_arch.png)
</div>
(y-axis is cross-entropy loss with 64-way candidate sampling loss. x-axis is batch #.)

## Optimizer

I didn't spend as much time evaluating optimizers as I would have liked. But SGD w/ Momentum outperformed both Adam and RMSProp for a variety of settings. Both Adam and RMSProp were significantly (about 2x) slower per epoch, but made no faster progress and SGD w/ Momentum.

TODO - Fire off a background job to check this now that other model parameters are settled. Perhaps include a plot with relative time.

## RNN Cell

I won't explain various RNN cell types here - [this post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) already provides a terrific explanation.

### GRU

The GRU cell appears more elegant than the LSTM cell. The same control is used for the input and forgetting gates. Also, the cell state and the hidden state are cleverly combined into one (i.e. no separate cell state). You would think this cell would run faster, and perhaps be easier to train. I found in TensorFlow that GRU ran **slower** than the LSTM. And, if anything, GRU was a bit more touchy to changes learning rate. 

[This paper](https://arxiv.org/pdf/1412.3555v1.pdf) does an in-depth comparison of GRU vs. LSTM. They found mixed performance results. However, their time per epoch is consistently faster with the GRU. So, perhaps there is an implementation issue with the TensorFlow **GRUCell** (or, alternatively, with the paper's LSTM cell)?

### Basic RNN

There was no big surprise here. This cell was very difficult to train even with a sequence length of 5. For example, in this training run the gradients exploded after about 25 epochs:

TO-DO: get screen shot

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![t-SNE Learning](assets/.gif)
</div>

## Learning Rate

## Gradient Norm Clipping

For LSTM Norm clipping wasn't helpful. There were cases with a low learning rate most of the activations would saturate. I expected adding norm clipping would solve this. It did slow the process, but the activations all headed to the rails. But really increasing the learning rate seemed to be the biggest help. 


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
