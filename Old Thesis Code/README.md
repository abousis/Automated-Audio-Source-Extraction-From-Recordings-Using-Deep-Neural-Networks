## Thesis Code
This is my diploma thesis code. For more info check my thesis in front page.

## Installation
Install the same packages as in front page. To evaluate the method install also the following using [pip](https://pip.pypa.io/en/stable/) 

```bash
pip install mir_eval
```

or if you are using [Anaconda](https://anaconda.org/)

```bash
conda install -c conda-forge mir_eval 
```

You will need also [h5py](https://www.h5py.org/) package to create the static dataset. Install with [pip](https://pip.pypa.io/en/stable/)

```bash
pip install h5py
```
or [Anaconda](https://anaconda.org/)

```bash
conda install -c anaconda h5py 
```

## Usage
To reproduce results, run scripts in the following order

```bash
python preprocessing_old.py
python autoencoder_old.py
python training_old.py
python postprocessing_old.py 
```

