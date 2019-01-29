visualize <- function(rules, W, W2) {
  firstNodeIds <- 1:dim(W)[1]
  firstNodeLabels <- paste(c(as.character(inspect(rules)$lhs)),"=>",c(as.character(inspect(rules)$rhs)))
  secondNodeIds <- 1:dim(W)[2]
  secondNodeLabels <- paste("Layer 2 Node",secondNodeIds)
  secondNodeIds <- secondNodeIds + dim(W)[1]
  thirdNodeIds <- 1:dim(W2)[2]
  thirdNodeLabels <- paste("Layer 3 Node",thirdNodeIds)
  thirdNodeIds <- thirdNodeIds + dim(W)[1] + dim(W)[2]
  fromEdges <- c(rep(1:dim(W)[1],dim(W)[2]),rep(1:dim(W2)[1],dim(W2)[2])+dim(W)[1])
  toEdges <- c(sort(rep(1:dim(W)[2],dim(W)[1])+dim(W)[1]),sort(rep(1:dim(W2)[2],dim(W2)[1])+dim(W)[1]+dim(W)[2]))
  edgeLabels <- c(c(W),c(W2))
  edgeLabels <- round(edgeLabels,2)
  fromEdges <- fromEdges[as.logical(edgeLabels)]
  toEdges <- toEdges[as.logical(edgeLabels)]
  edgeLabels <- edgeLabels[as.logical(edgeLabels)]
  nodeIds <- c(firstNodeIds,secondNodeIds,thirdNodeIds)
  nodeLabels <- c(firstNodeLabels,secondNodeLabels,thirdNodeLabels)
  nodesUsed <- unique(c(toEdges,fromEdges))
  nodesToKeep <- nodeIds %in% nodesUsed
  nodeIds <- nodeIds[nodesToKeep]
  nodeLabels <- nodeLabels[nodesToKeep]
  nodes <- data.frame(id = nodeIds, title=nodeLabels)
  edges <- data.frame(from = fromEdges, to = toEdges, label = edgeLabels)
  visNetwork(nodes, edges, width = "100%", height = "100%") %>% 
    visEdges(arrows="to", color=list(highlight="red",hover="red")) %>% 
    visHierarchicalLayout(direction="LR", sortMethod = "directed") %>%
    visOptions(highlightNearest = list(enabled =TRUE, degree = 1))
}