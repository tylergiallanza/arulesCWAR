# Classification Based on Weighted Association Rules

This R package 
implements the Classification Based on Weighted Association Rules (CWAR) algorithm. This algorithm is described in a forthcoming publication, Giallanza and Hahsler 2019. The package utilizes rtensorflow and arulesCBA.


## Installation

__Stable CRAN version:__ install from within R with
```R
install.packages("arulesCWAR")
```
__Current development version:__ 
```R 
library("devtools")
install_github("tylergiallanza/arulesCWAR")
```

## Usage

```R
data(Adult)
cwar_params <- list(support=0.3,confidence=0.5,weight_initialization='confidence',
                    loss='cross',regularization='l1',optimizer='sgd',epoch=2,batch_size=16,learning_rate=0.001,regularization_weights=list(l1=0.01))
## Not run:
classifier <- CWAR('income~.',Adult,cwar_params,0)
classifier
predict(classifier, head(Adult))
## End(Not run)
```