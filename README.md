# A set of datasets for teaching purposes

## Zaragoza contamination

Station Location and contaminant data. Based on Zaragoza open data.

## Wine Quality - Red (UCI)

https://archive-beta.ics.uci.edu/dataset/186/wine+quality

Regression - 11 continuous features

```
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/rmcantin/datasets/refs/heads/main/concrete.csv", sep=";")

X = df.drop(columns=["quality"])
y = df["quality]"
```

## Concrete Compressive Strength (UCI)

https://archive-beta.ics.uci.edu/dataset/165/concrete+compressive+strength

Regression - 8 continous features

```
import pandas as pd

df = pd.read_csv(https://raw.githubusercontent.com/rmcantin/datasets/refs/heads/main/concrete.csv)

X = df.drop(columns=["strength"])
y = df["strength"]
```

## Year Prediction - Million Songs Database (UCI)

https://samyzaf.com/ML/song_year/song_year.html

Regression - 90 features

```
import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/rmcantin/datasets/refs/heads/main/YearPredictionMSD.csv')

X = data.to_numpy()[:,1:]
y = data.to_numpy()[:,0]
```

## Weight Data - by Philipp Hennig

https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/lehre/probabilistic-machine-learning/

```
import numpy as np
import scipy as sp

data=sp.io.loadmat('weightdata_clean.mat')
X = np.atleast_2d(data['dat'][:,0]).T
Y = np.atleast_2d(data['dat'][:,1]).T
```