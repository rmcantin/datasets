# A set of datasets for teaching purposes

## Zaragoza contamination

## Wine Quality - Red (UCI) https://archive-beta.ics.uci.edu/dataset/186/wine+quality

Regression - 11 continuous features

```
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

X = df.drop(columns=["quality"])
y = df["quality]"
```

## Concrete Compressive Strength (UCI) https://archive-beta.ics.uci.edu/dataset/165/concrete+compressive+strength

Regression - 11 continous features

```
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)

X = df.drop(columns=["strength"])
y = df["strength"]
```

# Weight Data - by Philipp Hennig
https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/lehre/probabilistic-machine-learning/

```
import numpy as np
import scipy as sp

data=sp.io.loadmat('weightdata_clean.mat')
X = np.atleast_2d(data['dat'][:,0]).T
Y = np.atleast_2d(data['dat'][:,1]).T
```