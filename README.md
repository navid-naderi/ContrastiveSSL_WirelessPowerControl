# Self-supervised contrastive representation learning for downlink power control in wireless interference networks

This is a repository containing the PyTorch implementation of a learning-based power control approach in wireless networks with limited labeled data using self-supervised learning. Contrastive learning is used to pre-train the backbone to create embeddings where similar wireless channel matrices end up close to each other in the embedding space. Supervised learning is subsequently used to train the power control regression head. This approach can significantly reduce the need for data labeling, i.e., deriving optimal power control decisions, and boost the performance given a limited labeling budget.

To run the code for a network with 6 transmitter-receiver pairs on a GPU, try the following command:

```
python main.py --n 6 --device cuda
```

Check `main.py` for other optional arguments, such as the number of labeled samples, neural network size, learning rate, etc.
