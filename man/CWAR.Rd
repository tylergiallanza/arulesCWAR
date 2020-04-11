\name{CWAR}
\alias{CWAR}
\alias{cwar}
\title{Classification Based on Weighted Association Rules Algorithm (CWAR)}
\description{
  Build a classifier based on association rules mined for an input dataset using the CWAR algorithm.
}
\usage{
CWAR(formula, data, support = 0.1, confidence = 0.5, 
  disc.method = "mdlp", balanceSupport = TRUE, 
  training_parameter = NULL, mining_parameter = NULL, mining_control = NULL, verbose = FALSE)
}

\arguments{
  \item{formula}{A symbolic description of the model to be fitted. Has to be of form \code{class ~ .}
  or \code{class ~ predictor1 + predictor2}.}
  \item{data}{A transactions object or a data.frame containing the training data.}
  \item{support}{Support threshold for mining CARs.}
  \item{confidence}{Confidence threshold for mining CARs.}
  \item{disc.method}{Discretization method used to discretize continuous variables if data is a data.frame.}
  \item{balanceSupport}{balanceSupport parameter passed to the \code{\link[arulesCBA]{mineCARs}} function.}
  \item{training_params}{List of parameters for the CWAR algorithm.
  \describe{
    \item{weight_initialization}{Weight initialization. Options are uniform, random, and confidence.}
    \item{loss}{Loss function. Options are cross for categorical-crossentropy loss and mse for mean squared error.}
    \item{l1, l2}{Regularization parameter for L1 and L2 regularization.}
    \item{optimizer}{Used optimizer. Options are sgd, adam, and adadelta.}
    \item{opt_params}{List with parameters for the optimizer. Adam optimizer has additional params specificed at params$adam_params of learning_rate, beta1, beta2, and epsilon. Adadelta optimizer has additional params specificed at params$adadelta_params of learning_rate, rho, and epsilon.}
    \item{batch_size}{The number of transactions to include in each minibatch.}
    \item{epoch}{The number of training epochs.}
    \item{learning_rate}{learning rate.}
}}
  \item{verbose}{Show progress information.}
}
\details{
  Builds a classifier from the given transactions. First, association rules are generated with the Apriori algorithm from the arules package.
  
  From these associate rules, the CWAR algorithm is applied to build a classifier. The algorithm uses the tensorflow package to iteratively update rule weights with stochastic gradient descent.
}
\value{
  Returns an object of class \code{\link{CWAR}} representing the trained classifier.
}

\author{Tyler Giallanza}
\seealso{
\code{\link{CBA}},
\code{\link[arules]{apriori}},
\code{\link[arules]{rules}},
\code{\link[arules]{transactions}}.
}
\examples{
data(iris)
classifier <- CWAR(Species ~ ., iris, 
  support = 0.1, confidence = 0.5, 
  training_parameter = list(groups = 0, l1 = 0.05, epochs = 100), 
  verbose = TRUE
)

classifier
inspect(rules(classifier))

pr <- predict(classifier, iris)
pr
table(pr, iris$Species)


data(Mushroom, package = "arules")
Mushroom
  
classifier <- CWAR(Class ~ ., Mushroom, support = 0.1, confidence = 0.5, mining_parameter = list(maxlen = 3),
  training_parameter = list(l1 = 0.01, groups = 10, epoch = 10), verbose = TRUE)

classifier
inspect(rules(classifier))

predict(classifier, head(Mushroom), type = "probability")
predict(classifier, head(Mushroom))
table(predict(classifier, Mushroom), as.vector(as(Mushroom[, c("Class=edible")], "matrix")))
} 
