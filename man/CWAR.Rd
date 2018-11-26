\name{CWAR}
\alias{CWAR}
\alias{cwar}
\title{Classification Based on Weighted Association Rules Algorithm (CWAR)}
\description{
  Build a classifier based on association rules mined for an input dataset using the CWAR algorithm.
}
\usage{
CWAR(formula, data, params = NULL, verbosity=0)
}

\arguments{
  \item{formula}{A symbolic description of the model to be fitted. Has to be of form \code{class ~ .}
  or \code{class ~ predictor1 + predictor2}.}
  \item{params}{List of parameters for the CWAR algorithm.}
  \item{data}{A transactions object containing the training data.}
  \item{verbosity}{Optional parameter to control the verbosity of the output.}
}
\details{
  Builds a classifier from the given transactions. First, association rules are generated with the Apriori algorithm from the arules package.
  
  From these associate rules, the CWAR algorithm is applied to build a classifier. The algorithm uses the tensorflow package to iteratively update rule weights with stochastic gradient descent.
}
\value{
  Returns an object of class \code{\link{CBA}} representing the trained classifier.
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
cwar_params <- list(support=0.3,confidence=0.5,weight_initialization='confidence',
                    loss='cross',regularization='l1',optimizer='sgd',epoch=2,batch_size=16,learning_rate=0.001,regularization_weights=list(l1=0.01))
classifier <- CWAR('income~.',Adult,cwar_params,0)
classifier
predict(classifier, head(Adult))
}
