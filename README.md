# WiFi-based Human Activity Recognition using Raspberry Pi

Code used for the experiments performed in our [2020 ICTAI submission](https://ieeexplore.ieee.org/abstract/document/9288199]).

Sections of this code were used in other experiments and are not used in this version. The version used for this submission should be run as-is with no code changes.

## Instructions:
 - Download our Pi 4 Activity Recognition dataset from [Zenodo](https://zenodo.org/record/5616432#.YxXszy8w29l).
 - Place the data folder in this directory.
 - Run using `main.py` or the included VSCode `.devcontainer` configuration.
    - Install dependencies if necessary via `pip install -r requirements.txt`
 - Run the experiment via `main.py`

## Results

The following results were achieved using the configuration parameters set for `newpi` in `config.json`.

```
Epoch 100/100
14/14 [==============================] - 1s 48ms/step - loss: 0.2284 - accuracy: 0.9266 - val_loss: 0.2829 - val_accuracy: 0.9059

[[155   0   0   0   0   0   0   0   0   0   0]
 [  0  93  34   0   0   0   0   0   0   0   0]
 [  0  80  52   1   0   0   0   1   0   0   0]
 [  0   0   0 126   0   0   0   1  15   0   0]
 [  0   0   0   0 145   0   0   0   0   0   0]
 [  0   0   0   0   0 158   0   0   0   0   0]
 [  0   0   0   0   0   0 143   0   0   0   0]
 [  0   0   0   0   0   0   0  96   0   0   0]
 [  0   0   0   0   6   0   0   0  60   0   0]
 [  0   0   0   0   0   0   0   0   0 155   0]
 [  0   0   0   0   0   0   0   0   0   0 145]]
               precision    recall  f1-score   support

      nothing       1.00      1.00      1.00       155
      standup       0.54      0.73      0.62       127
      sitdown       0.60      0.39      0.47       134
   getintobed       0.99      0.89      0.94       142
         cook       0.96      1.00      0.98       145
washingdishes       1.00      1.00      1.00       158
   brushteeth       1.00      1.00      1.00       143
        drink       0.98      1.00      0.99        96
       petcat       0.80      0.91      0.85        66
     sleeping       1.00      1.00      1.00       155
         walk       1.00      1.00      1.00       145

     accuracy                           0.91      1466
    macro avg       0.90      0.90      0.90      1466
 weighted avg       0.91      0.91      0.90      1466
```

## License

The code in this project is licensed under MIT license. If you are using this codebase for any research or other projects, I would greatly appreciate if you could cite this repository or one of my papers.

a) "G. Forbes. CSIKit: Python CSI processing and visualisation tools for commercial off-the-shelf hardware. (2021). https://github.com/Gi-z/CSIKit."

b) "Forbes, G., Massie, S. and Craw, S., 2020, November. 
      WiFi-based Human Activity Recognition using Raspberry Pi. 
      In 2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI) (pp. 722-730). IEEE."

  ```
  @electronic{csikit:gforbes,
      author = {Forbes, Glenn},
      title = {CSIKit: Python CSI processing and visualisation tools for commercial off-the-shelf hardware.},
      url = {https://github.com/Gi-z/CSIKit},
      year = {2021}
  }

  @inproceedings{forbes2020wifi,
    title={WiFi-based Human Activity Recognition using Raspberry Pi},
    author={Forbes, Glenn and Massie, Stewart and Craw, Susan},
    booktitle={2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI)},
    pages={722--730},
    year={2020},
    organization={IEEE}
  }
  ```