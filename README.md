# Self-supervised contrastive representation learning for downlink power control in wireless interference networks (accepted to IEEE ICASSP 2021)

This is a repository containing the PyTorch implementation of a learning-based power control approach in wireless networks with limited labeled data using self-supervised learning. Contrastive learning is used to pre-train the backbone to create embeddings where similar wireless channel matrices end up close to each other in the embedding space. Supervised learning is subsequently used to train the power control regression head. This approach can significantly reduce the need for data labeling, i.e., deriving optimal power control decisions, and it considerably boosts the performance given a limited labeling budget.

To run the code for a network with 8 transmitter-receiver pairs on a GPU, try the following command:

```
python3 main.py --n 8 --device cuda:0
```

Check `main.py` for other optional arguments, such as the number of labeled samples, neural network size, learning rate, etc.

Moreover, the Jupyter notebook `SSL_2DEmbedding_Clusters.ipynb` can be used to plot the 2-dimensional embeddings for networks with 3 transmitter-reciever pairs before and after the self-supervised pre-training phase.

##

If you use this repository in your work, please cite the accompanying paper using the BibTeX citation below:

```
@article{naderializadeh2020contrastive,
  title={Contrastive Self-Supervised Learning for Wireless Power Control},
  author={Naderializadeh, Navid},
  journal={arXiv preprint arXiv:2010.11909},
  year={2020}
}
```