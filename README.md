# praca-magisterska
Build project with command:

```
mvn clean package
```

To run application you should specify:
- kind of task to execution (evaluation, recommendation)
- dataset
- model class
- algorithm name with packages path
- neighborhood size (in case of k-NN algorithm) 
- test and training datasets (in case of evaluation task)

In datasets directory there are two dataset used for master thesis experiments - MovieLens1M and Book-Crossing. There are also training and test datasets created on the basis of this data.

To run experiments you can use Python scripts in mahout-scripts and lenskit-scripts directory.
