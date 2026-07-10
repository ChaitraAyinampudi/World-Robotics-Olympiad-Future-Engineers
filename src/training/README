# DonkeyCar Model Training

This folder contains the files used to train our autonomous driving models.

## Files

### `train.py`

Starts the DonkeyCar training process. It loads configuration, reads recorded tub data, and saves a trained model.

Example:

```bash
python3 train.py --tubs data/ --model models/mypilot.h5
```

### `config.py`

Contains DonkeyCar's main default settings, including:

* Camera resolution
* Model type
* Training and validation split
* Batch size
* Epoch limit
* Early stopping

### `myconfig.py`

Contains our custom training settings. Active values in this file override matching values in `config.py`.

## Current Settings

| Setting             |      Value |
| ------------------- | ---------: |
| AI framework        | TensorFlow |
| Model type          |     Linear |
| Batch size          |        512 |
| Maximum epochs      |        200 |
| Early-stop patience |         15 |
| Minimum improvement |     0.0002 |
| Training split      |        80% |
| Validation split    |        20% |


## Files

training/
├── README.md
├── train.py
├── config.py
└── myconfig.py
