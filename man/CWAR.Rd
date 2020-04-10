\name{CWAR}
\alias{CWAR}
\alias{cwar}
\title{Classification Based on Weighted Association Rules Algorithm (CWAR)}
\description{
  Build a classifier based on association rules mined for an input dataset using the CWAR algorithm.
}
\usage{
CWAR(formula, data, mining_params, training_params, opt_params)
}

\arguments{
  \item{formula}{A symbolic description of the model to be fitted. Has to be of form \code{class ~ .}
  or \code{class ~ predictor1 + predictor2}.}
  \item{data}{A transactions object containing the training data.}
  \item{mining_params}{Parameters for the apriori algorithm. A list with the following elements:
    \describe{
  \item{support}{Support control for the apriori algorithm.}
  \item{confidence}{Confidence control for the apriori algorithm.}
  }}
  \item{training_params}{List of parameters for the CWAR algorithm.
  \describe{
  \item{weight_initialization}{Weight initialization scheme for CWAR - options are uniform, random, and confidence.}
  \item{oss}{Loss function for CWAR - options are cross for categorical-crossentropy loss and mse for mean squared error.}
  \item{optimizer}{Optimizer for CWAR - options are sgd, adam, and adadelta. Adam optimizer has additional params specificed at params$adam_params of learning_rate, beta1, beta2, and epsilon. Adadelta optimizer has additional params specificed at params$adadelta_params of learning_rate, rho, and epsilon.}
  \item{regularization}{Regularization scheme for CWAR - options include none, l1, l2, and elastic. Additional params specified at params_regularization_weights include l1 and l2.}
  \item{epoch}{The number of epochs to use when training CWAR.}
  \item{batch_size}{The number of transactions to include in each minibatch.}
}}
  \item{opt_params}{Parameters for the optimizer of CWAR.}
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
data(Adult)

classifier <- CWAR(income ~ ., Adult, support=0.3, confidence=0.5, 
  training_params = list(loss='cross'), 
  opt_params = list(0.1)
)

classifier
predict(classifier, head(Adult), type = "probability")
predict(classifier, head(Adult))


# with no grouping
classifier <- CWAR(income ~ ., Adult, 
  support=0.3, confidence=0.5, 
  training_params = list(loss = 'cross', deep=0L, l1 = .001), 
  opt_params = list(0.1)
)

classifier
predict(classifier, head(Adult), type = "probability")
predict(classifier, head(Adult))

inspect(sort(rules(classifier), by = "weight"))

# use non transaction data directly
data(iris)
classifier <- CWAR(Species ~ ., iris, 
  support=0.01, confidence=0.5, 
  training_params = list(loss = 'cross', deep=0L, l1 = 0.2), 
  opt_params = list(0.1)
)

classifier
inspect(sort(rules(classifier), by = "weight"))

pr <- predict(classifier, iris)
pr
table(pr, iris$Species)
} 
