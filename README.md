# WiFi-based Human Activity Recognition using Raspberry Pi

Code used for the experiments performed in our [2020 ICTAI submission](https://ieeexplore.ieee.org/abstract/document/9288199]).

Sections of this code were used in other experiments and are not used in this version. The version used for this submission should be run as-is with no code changes.

## Instructions:
 - Download our Pi 4 Activity Recognition dataset from [Zenodo](https://zenodo.org/record/5616432#.YxXszy8w29l).
 - Place the data folder in this directory.
 - Run using `main.py` or the included VSCode `.devcontainer` configuration.
    - Install dependencies if necessary via `pip install -r requirements.txt`
 - Run the experiment via `main.py`

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