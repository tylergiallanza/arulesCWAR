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
  \item{mining_params}{Parameters for the apriori algorithm.}
  \item{mining_params$support}{Support control for the apriori algorithm.}
  \item{mining_params$confidence}{Confidence control for the apriori algorithm.}
  \item{training_params}{List of parameters for the CWAR algorithm.}
  \item{training_params$weight_initialization}{Weight initialization scheme for CWAR - options are uniform, random, and confidence.}
  \item{training_params$loss}{Loss function for CWAR - options are cross for categorical-crossentropy loss and mse for mean squared error.}
  \item{training_params$optimizer}{Optimizer for CWAR - options are sgd, adam, and adadelta. Adam optimizer has additional params specificed at params$adam_params of learning_rate, beta1, beta2, and epsilon. Adadelta optimizer has additional params specificed at params$adadelta_params of learning_rate, rho, and epsilon.}
  \item{training_params$regularization}{Regularization scheme for CWAR - options include none, l1, l2, and elastic. Additional params specified at params_regularization_weights include l1 and l2.}
  \item{training_params$epoch}{The number of epochs to use when training CWAR.}
  \item{training_params$batch_size}{The number of transactions to include in each minibatch.}
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
mining_params <- list(support=0.3,confidence=0.5)
training_params <- list(loss='cross')
opt_params <- list(0.1)
## Not run:
classifier <- CWAR('income~.',Adult,mining_params,training_params,opt)_params)
classifier
predict(classifier, head(Adult))
## End(Not run)
}
