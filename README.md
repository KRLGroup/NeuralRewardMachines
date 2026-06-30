# Neural Reward Machines
Official repository for the paper "Neural Reward Machines. Elena Umili, Francesco Argenziano and Roberto Capobianco. 27th European Conference in Artificial Intelligence (ECAI 2024)".

## Requirements
- pytorch
- gym
- pygame

## How reproduce the experiments
To reproduce the experiments in the paper run the script ```experiments.py```

The script uses some flags, run ``` python experiments.py --helpfull``` to see the full list of parameters used.

```
       USAGE: experiments.py [flags]
flags:

experiments.py:
  --ENV: Environment to test, one in ['map_env', 'image_env'], default= 'map_env'
    (default: 'map_env')
  --LOG_DIR: path where to save the results, default='Results/'
    (default: 'Results/')
  --METHOD: Method to test, one in ['rnn', 's4', 'nrm', 'rm'], default= 'rnn'
    (default: 'rnn')
  --NUM_EXPERIMENTS: num of runs for each test, default= 5
    (default: '5')
    (an integer)
  --REWARD: reward scheme of the Moore machine, one in ['distance',
    'acceptance', 'three_value_acceptance'], default= 'three_value_acceptance'
    (default: 'three_value_acceptance')
  --[no]USE_REPLAY_BUFFER: for 'nrm', train the grounder via balanced replay
    buffers (True) or on the worst recent trajectories (False), default= False
    (default: 'false')
```
## Citations
```
@inproceedings{DBLP:conf/ecai/UmiliAC24,
  author       = {Elena Umili and
                  Francesco Argenziano and
                  Roberto Capobianco},
  editor       = {Ulle Endriss and
                  Francisco S. Melo and
                  Kerstin Bach and
                  Alberto Jos{\'{e}} Bugar{\'{\i}}n Diz and
                  Jose Maria Alonso{-}Moral and
                  Sen{\'{e}}n Barro and
                  Fredrik Heintz},
  title        = {Neural Reward Machines},
  booktitle    = {{ECAI} 2024 - 27th European Conference on Artificial Intelligence,
                  19-24 October 2024, Santiago de Compostela, Spain - Including 13th
                  Conference on Prestigious Applications of Intelligent Systems {(PAIS}
                  2024)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {392},
  pages        = {3055--3062},
  publisher    = {{IOS} Press},
  year         = {2024},
  url          = {https://doi.org/10.3233/FAIA240847},
  doi          = {10.3233/FAIA240847},
  timestamp    = {Fri, 25 Oct 2024 12:13:46 +0200},
  biburl       = {https://dblp.org/rec/conf/ecai/UmiliAC24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```
@inproceedings{UmiliVisualRewardMachines,
  author       = {Elena Umili and
                  Francesco Argenziano and
                  Aymeric Barbin and
                  Roberto Capobianco},
  editor       = {Artur S. d'Avila Garcez and
                  Tarek R. Besold and
                  Marco Gori and
                  Ernesto Jim{\'{e}}nez{-}Ruiz},
  title        = {Visual Reward Machines},
  booktitle    = {Proceedings of the 17th International Workshop on Neural-Symbolic
                  Learning and Reasoning, La Certosa di Pontignano, Siena, Italy, July
                  3-5, 2023},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3432},
  pages        = {255--267},
  publisher    = {CEUR-WS.org},
  year         = {2023},
  url          = {https://ceur-ws.org/Vol-3432/paper23.pdf},
  timestamp    = {Tue, 11 Jul 2023 17:14:10 +0200},
  biburl       = {https://dblp.org/rec/conf/nesy/UmiliABC23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
