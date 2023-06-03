# DSL
This is our pytorch implemention for our DSL:

## Environment Requirement

The required packages are as follows:
+ python == 3.6.13
+ pytorch == 1.10.2
+ dgl-cuda11.3 == 0.8.1
+ numpy == 1.19.2
+ pandas == 1.1.5

## Datasets

You can download original dataset from the following links:
[Movielens-1M]: https://grouplens.org/datasets/movielens/1m/
[CiteULike]: https://github.com/js05212/citeulike-t
[Foursquare]: https://sites.google.com/site/yangdingqi/home/foursquare-dataset
[Yelp2018]: https://www.yelp.com/dataset
[Gowalla]: https://snap.stanford.edu/data/loc-gowalla.html
[Amazon-book]: https://jmcauley.ucsd.edu/data/amazon/

We also provide the processed dataset for Movielens-1M and Yelp2018 in [data file](./data)

## Parameters
Key parameters are mentioned in our paper. More related parameters are all listed in the [parse_args](main.py) or [parse_args](config.py)

## Quick-Start

We implemented DSL on the MultVAE model, which you can run directly in [model file](./MultVAE)

  ```
  CUDA_VISIBLE_DEVICES=$1 python -u main.py --cuda --epoch 200 \
                                            --data ml-1m --save d0.5_model_ml1m.pt \
                                            --sparse --density 0.5 --update_frequency 100 \
                                            --batch_size 16
  ```

We implemented DSL on the NeuMF model, which you can run directly in [model file](./NeuMF):

  ```
  CUDA_VISIBLE_DEVICES=$1 python -u main.py --num-epochs 200 --batch-size 512  \
                                            --sparse --density 0.5 --update_frequency 2000 \
                                            --dataset ml-1m
  ```

We implemented DSL on the LightGCN model, which you can run directly in [model file](./LightGCN):

  ```
  CUDA_VISIBLE_DEVICES=$1 python -u main.py --dataset yelp2018 --epoch 1000 \
                                            --embedding_size 64 --num_layer 3 --batch_size 1024 \
                                            --density 0.5 --sparse --update_interval 3000
  ```

## Data processing

The processing for Yelp2018, Gowalla and Amazon-book: 
Follow the baseline model [LightGCN](https://arxiv.org/pdf/2002.02126.pdf) and [UltraGCN](https://arxiv.org/pdf/2110.15114.pdf)

The processing for CiteULike and Foursquare:
Please remove users having fewer than 5 (CiteULike) and 20 interactions (FourSquare) and remove items having fewer than 10 interactions(FourSquare).

More detailed processing and spliting for data please refer to [data file](./data/process.py)