# ITU Beam Selection Challenge

###### This repo contains the code for the paper Federated mmWave Beam Selection Utilizing LIDAR Data

## Authors

* Dr. Mahdi Boloursaz Mashhadi (PostDoc Team Leader), Imperial College London, Email: m.boloursaz-mashhadi@imperial.ac.uk
* Mr. Mikolaj Jankowski, Imperial College London, Email: mikolaj.jankowski17@imperial.ac.uk
* Mr. Tze-Yang Tung, Imperial College London, Email: tze-yang.tung14@imperial.ac.uk
* Mr. Szymon Kobus, Imperial College London, Email: szymon.kobus17@imperial.ac.uk
* <b>Supervisor</b>: Prof. Deniz Gunduz, Imperial College London, Email: d.gunduz@imperial.ac.uk

## Requirements installation
I order to run our code, you need PyTorch and Numpy. In order to install them, simply run:
```
$ pip install -r requirements.txt
```
Since our code usues GPU acceleration, having CUDA capable device would be highly recommended, along with CUDA toolkit v10.2.

For your convenience, we also provide a TensorFlow implementation of our code. It can be accessed through this link: https://github.com/galidor/ITU_Beam_Selection_TF.git

## Source files:
There are 4 Python files in this repo, requirements.txt, and this README file:
* models.py - definition of our model called Baseline2D.
* dataset.py - dataset preprocessing required by PyTorch, along with our own preprocessing steps that transform 3D baseline data to 2D representations. Please note, even though we don't use the baseline features, it is not required to perform any preprocessing, as our code does it directly before the training/testing with a very small computational overhead.
* beam_train_model.py - code that is executed to perform full training of our network.
* beam_test_model.py - code that enables to test previously trained networks and generate .csv files with the predictions.

## Dataset
In this work we use Raymobtime<sup>2</sup> dataset, which is a collection of realistic ray-tracing data obtained by simulating traffic in environment highly inspired by real world data. It utilizes SUMO for mobility simulations, Insite for ray-tracing, Cadmapper and Open Street Map for importing realistic outdoor scenarios. The dataset is divided into smaller sets, with different frequencies considered, various number of receivers, and environments. In this work, we trained on the s008 dataset (we combined both training and validation subsets) and validated on s009. 

## Model training
In order to train our network, you need to run the following command:

```
$ python beam_train_model.py --lidar_training_data <path to the LIDAR baseline training .npz file>
                             --beam_training_data <path to the baseline training beam .npz file>
                             --lidar_validation_data <path to the LIDAR baseline validation .npz file>
                             --beam_validation_data <path to the baseline validation beam .npz file>
                             --model_path <path, where your model will be stored>
```

Please note, that if the validation data are not provided, training data will be used as the validation set. If you provide multiple .npz files, they will be combined into larger dataset.

## Evaluation
In order to evaluate pretrained model, simply run evaluate.py, as follows:
```
$ python beam_test_model.py --lidar_test_data <path to the LIDAR baseline test .npz file>
                            --beam_test_data <path to the baseline test beam .npz file>
                            --model_path <path, where your model is stored>
                            --preds_csv_path <path, where you want your .csv file with the predictions to be stored>
```

## Questions?
If you have any further questions related to this repo, feel free to contact me at mikolaj.jankowski17@imperial.ac.uk or raise an Issue within this repo. I'll do my best to reply as soon as possible.
   
## References
1. M. B. Mashadi, M. Jankowski, T-Y. Tung, S. Kobus, D. Gunduz, “Federated mmWave Beam Selection Utilizing LIDAR Data”, arXiv:2102.02802, link: https://arxiv.org/abs/2102.02802
2. ITU AI/ML in 5G Challenge website: https://www.itu.int/en/ITU-T/AI/challenge/2020/Pages/default.aspx
3. Raymobtime dataset: https://www.lasse.ufpa.br/raymobtime/
4. A. Klautau, P. Batista, N. González-Prelcic, Y. Wang and R. W. Heath Jr., “5G MIMO Data for Machine Learning: Application to Beam-Selection using Deep Learning” in 2018 Information Theory and Applications Workshop (ITA).
