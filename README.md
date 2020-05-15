# insulin-intake

Python multivariate linear regression algorithm which helps users with diabetes predict the amount of insulin to take based on glucose level, carbs intake, units of carbs, excersice done or not (defined by, 1 done, 0 not done) and time (in 24 hour format without colon).

##Getting started

These instructions will get you a copy of the project up and running on your local machine for training and testing purposes.

###Prerequisites

* You should have python 3.x.x installed (You can install it [here](https://www.python.org/downloads/))
* You should have pip installed (You can install it [here](https://pip.pypa.io/en/stable/installing/)) 

###Installing libraries needed

The following libraries are needed in order tu run the algorithm (If you already have them installed we recommend you update them to their respective latest version to avoid any issues).

```
pip3 install numpy
```

```
pip3 install pandas
```

###Using your own dataset

If you wish to use your own dataset you could do so by adding it under the **datasets** folder.
* Please make sure to use a csv format.
* The order of the features doesn't affect the functionality of the algorithm as long as you train and test with the same format.

Under the **sources** folder, open *multivariate-linear-regression.py* and on line 29 change the first parameter of
```
uf.load_data_multivariate('./datasets/training-data-multivariate.csv', 'training')
```
to
```
uf.load_data_multivariate('./datasets/name-of-your-dataset.csv', 'training')
```

###Authors

* [Mauricio A. De León Cárdenas](https://github.com/mauriciodeleonc) 505597
* [Juan M. Álvarez Sánchez](https://github.com/jm_alvarezs) xxxxxx
* [Viviana V. Gómez Martínez](https://github.com/mauriciodeleonc) xxxxxx
* [Orlando X. Torres Guerra](https://github.com/mauriciodeleonc) xxxxxx

###Honor code
We hereby declare that we have worked in this project with academic integrity.
