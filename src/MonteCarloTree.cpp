#include "MonteCarloTree.hpp"

CudaMctsCheckers::MonteCarloTree::MonteCarloTree() { root = new MonteCarloTreeNode(); }

CudaMctsCheckers::MonteCarloTree::~MonteCarloTree() { delete root; }

CudaMctsCheckers::MonteCarloTreeNode::MonteCarloTreeNode(
    CudaMctsCheckers::MonteCarloTreeNode *parent
)
{
    this->parent = parent;
}

CudaMctsCheckers::MonteCarloTreeNode::~MonteCarloTreeNode()
{
    for (auto &child : children) {
        delete child.second;
    }
}
