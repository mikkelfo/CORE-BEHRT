import torch
from torch.nn import NLLLoss
from torch import softmax
from typing import List, Tuple


class CE_FlatSoftmax(torch.nn.Module):
    """Computing a cross entropy loss on a tree with flat softmax on leaf nodes.
    Loss is only computed on target level."""
    def __init__(self, leaf_nodes) -> None:
        super(CE_FlatSoftmax, self).__init__()  
        self.leaf_nodes = leaf_nodes
        self.nllloss = NLLLoss()

    def forward(self, leaf_logits, y_true_enc):
        """Selects leaf probabilities for a given target tensor.
        Args:
            leaf_logits (torch.tensor): Logits (batchsize, num_leaf_nodes)
            y_true_enc (torch.tensor): Target vector (batchsize, seq_len, levels)
            leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
        Returns:
            Cross entropy loss
        """
        leaf_probs = softmax(leaf_logits, dim=-1)
        selected_leaf_probs = self.get_leaf_node_probabilities(leaf_probs, y_true_enc)
        # print(selected_leaf_probs)
        log_probs = torch.log(selected_leaf_probs)
        log_probs = log_probs.flatten() # batchsize * seq_len
        loss = self.nllloss(log_probs.unsqueeze(-1), torch.zeros_like(log_probs, dtype=torch.int64))
        return loss

    def get_leaf_node_probabilities(self, leaf_probs:torch.tensor, y_true_enc: torch.tensor):
        """Selects leaf probabilities for a given target tensor.
        Args:
            leaf_probs (torch.tensor): Probabilities (batchsize, num_leaf_nodes)
            y_true_enc (torch.tensor): Target vector (batchsize, seq_len, levels)
            leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
        Returns:
            torch.tensor: Selected leaf probabilities (batchsize, seq_len)"""
        # we want to match all the leaf nodes with a target, e.g. target: 1,2,0 should select 1,2,1 and 1,2,2
        zeros_mask = y_true_enc == 0
        leaf_mask = (self.leaf_nodes == y_true_enc[:, :, None, :] )| zeros_mask[:, :, None,:] # select all leaf nodes that match the target
        leaf_mask = leaf_mask.all(dim=-1).to(torch.int16)

        leaf_probs = leaf_probs[:,None,:].expand(leaf_mask.shape) # batch, seq_len, num_leafes

        selected_leaf_probs = leaf_probs * leaf_mask
        selected_leaf_probs = selected_leaf_probs.sum(dim=-1)
        return selected_leaf_probs


# TODO: implement CE_FlatSoftmax as special case of CE_FlatSoftmax_MOP, by setting all level contribution except the target level to 0
class CE_FlatSoftmax_MOP(torch.nn.Module):
    """
    Multiple Operating Points Cross Entropy Loss with flat softmax on leaf nodes.
    
    Cross entropy loss, where we normalize probabilities over all leaf nodes, and susequently compute a loss on every level of the tree. The result is a weighted sum of the losses on each level.
    For the target level, and above the target level the loss is computed as usual (assigning probability 1 to the target node, and 0 to all other nodes).
    For levels below the target level (if target has descendants on that level) probabilities are split equally among all descendandts.

    The call method takes the predicted leaf probabilities (batchsize, seq_len, num_leaf_nodes) and the target vector (batchsize, seq_len, levels), and returns the loss.
    """
    def __init__(self, leaf_nodes: torch.tensor, trainable_weights=False, ignore_index=-100) -> None:
        """Args:
        leaf_nodes (torch.tensor): Leaf nodes (num_leaf_nodes, levels)
        trainable_weights (bool, optional): Whether to train the level weights. Defaults to False."""
        super(CE_FlatSoftmax_MOP, self).__init__()  # Add this line
        self.ignore_index = ignore_index
        self.leaf_nodes = leaf_nodes
        self.lvl_mappings = self.get_level_mappings()
        self.lvl_sel_mats, self.nodes = self.construct_level_selection_mats_and_graph() # when leaf_probs multiplied fir lvl_sel_mat from the left -> probs on that level
        self.weights = self.initialize_geometric_weights()
        if trainable_weights:
            self.weights.requires_grad = True

    def forward(self, leaf_logits:torch.tensor, y_true_enc:torch.tensor)->float:
        loss = 0 # we will sum the losses on each level
        # predictions is in shape (batchsize, seq_len, num_leaf_nodes), we reshape to (batchsize * seq_len, num_leaf_nodes)
        leaf_logits = leaf_logits.reshape(-1, leaf_logits.shape[-1])
        predicted_leaf_probs = softmax(leaf_logits, dim=-1)
        for level, mat in enumerate(self.lvl_sel_mats):
            pred_probs_lvl = mat @ predicted_leaf_probs.T # shape (num_nodes, batchsize * seq_len)
            target_probs_lvl, pad_mask = self.construct_target_probability_mat(y_true_enc, level) # we can already pass it as a target!
            loss_lvl = self.categorical_cross_entropy(pred_probs_lvl.T, target_probs_lvl.T, pad_mask.T)         
            loss += loss_lvl * self.weights[level]
        return loss
    
    @staticmethod
    def categorical_cross_entropy(y_pred, y_true, mask=None):
        """Takes predicted and true probabilities and returns categorical cross entropy."""
        y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9) # we clip to avoid log(0)
        return -(y_true * torch.log(y_pred))[mask].sum()/y_true.shape[0] # we divide by n_samples to get mean

    def initialize_geometric_weights(self):
        """We initialize weights as e**(-i)"""
        return torch.nn.Parameter(torch.exp(-torch.arange(len(self.lvl_sel_mats))))

    def get_level_mappings(self):
        """Returns a dictionary, where the key is the level and the value is a tensor
        with the corresponding indices for this level.
        """
        lvl_mappings = []
        for i in range(len(self.leaf_nodes[0])):
            leaf_nodes_part = self.leaf_nodes[:,:i+1]
            _, unique_indices = torch.unique(leaf_nodes_part, dim=0, return_inverse=True) # we can use these to enumerate the target, in order to access the correct prob
            lvl_mappings.append([leaf_nodes_part, unique_indices])
        return lvl_mappings

    def construct_level_selection_mats_and_graph(self)->List[torch.tensor]:
        """For every level constructs a matrix, such that when multiplied from the left
        with leaf probabilities, returns probabs for every node on this level. 
        Matrices are returned as list. 
        """
        mats = []
        nodes = []
        for level in range(len(self.leaf_nodes[0])):
            leaf_nodes_part = self.leaf_nodes[:,:level+1]
            # remove zero nodes (below leaf)
            unique_nodes, unique_indices = torch.unique(leaf_nodes_part, dim=0, return_inverse=True) # we can use these to enumerate the target, in order to access the correct prob
            # Map each unique row to an integer based on its position in the sorted unique tensor
            mat = self.create_leaf_selection_matrix(unique_indices)
            # zero means we are below the leaf, so we set the corresponding row to zero
            zero_mask = leaf_nodes_part[:,-1]==0
            mat[:,zero_mask] = 0 # set elements to zero if the node ends with zero (node is on a higher level)
            all_zero_mask = mat.sum(dim=1)!=0

            mat = mat[all_zero_mask] # remove zero rows (corresponding to nodes from other levels)
            unique_nodes = unique_nodes[all_zero_mask] # remove zero rows (corresponding to nodes from other levels)
            
            mats.append(mat)
            nodes.append(unique_nodes)
        return mats, nodes

    def construct_target_probability_mat(self, y_true_enc:torch.tensor, level:int)->Tuple[torch.tensor,torch.tensor]:
        """For a specific level, construct a matrix of target probabilities.
        If target is given on level: one hot encoding
        elif target is given on a higher level: probabilities split equally among all children
        else: zero
        Parameters: 
            nodes: list of tensors, where each tensor is a list of nodes for a specific level
            y_true_enc: target tensor of shape (batchsize, seq_len, levels) 
        Returns:
            A: matrix of shape (batchsize x seq_len, num_nodes) where each row corresponds to a target and each column to a node on the given level"""
        nodes_lvl = self.nodes[level]
        num_nodes = nodes_lvl.shape[0]
        y_flat = torch.flatten(y_true_enc[:,:,:level+1], start_dim=0, end_dim=1) # flatten batch dim to simplify
        in_tree_mask = ((y_flat==nodes_lvl[:,None]).all(dim=-1).any(dim=0) | (y_flat==self.ignore_index).all(dim=-1)) | (y_flat==0).any(dim=-1) # we check that all targets are in the tree
        if not in_tree_mask.all():
            raise ValueError(f"Target node {y_flat[~in_tree_mask]} is not in the tree") # check that all targets are in the nodes
        pad_mask = self.get_mask(y_flat, level) # we get a (padding) mask for the current level
        probability_matrix = torch.zeros((num_nodes, y_flat.shape[0])) # we populate this matrix num_nodes x num_targets
        target_mask = self.get_target_mask(nodes_lvl, y_flat) # mask nodes that match target, shape: num_nodes x num_targets x levels

        target_mask_exact = target_mask.all(dim=2) # mask nodes that match target exactly
        probability_matrix[target_mask_exact] = 1
        target_child_mask = self.get_target_child_mask(target_mask, y_flat)
        
        child_probabilities = self.get_child_probabilities(target_child_mask)

        probability_matrix[target_child_mask] = child_probabilities[target_child_mask]
        return probability_matrix, pad_mask

    def get_mask(self, y_flat:torch.tensor, level:int):
        """Returns a mask for the current level, such that we can ignore padding tokens."""
        return (y_flat[:,:level+1] != self.ignore_index).all(dim=1)
    @staticmethod
    def create_leaf_selection_matrix(indices):
        """This function takes an array of integers and 
        returns a matrix where each row is a one-hot encoded version of the input.
        e,g, [1,1,1,2,2] -> [[1,1,1,0,0],[0,0,0,1,1]]]"""
        unique_values = torch.unique(indices)
        mask = (indices.unsqueeze(0) == unique_values.unsqueeze(1)).float()
        return mask
    @staticmethod
    def get_target_child_mask(target_mask, y_flat):
        """Returns a mask that is true if the node is a child of the target"""
        target_mask_parent = target_mask[:,:,:-1].all(dim=2) & (~target_mask[:,:,-1])  # mask nodes that match target up to the last level 
        target_mask_parent = target_mask_parent & (y_flat[:,-1]==0) # on the last level the target is zero
        return target_mask_parent
    @staticmethod
    def get_child_probabilities(target_child_mask):
        """Child probabilities are computed as 1/number of children"""
        return (1 / target_child_mask.sum(dim=0).float()).repeat(target_child_mask.shape[0], 1)
    @staticmethod
    def get_target_mask(nodes_lvl, y_flat):
        """Returns a mask of shape num_nodes x num_targets x n_levels that is true if the node matches the target"""
        return nodes_lvl[:,None,:]==y_flat