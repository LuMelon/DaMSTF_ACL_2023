import dgl, sys, math
sys.path.append("..")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from typing import List
from torch.nn import init

class GraphConv(nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        A = graph.adjacency_matrix().to_dense().to(feat.device)
        if self._norm:
            norm = torch.pow(graph.in_degrees().float(), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm
        rst = A.mm(feat).mm(self.weight)
        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class GraphConvV2(nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConvV2, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, A_Mtx, feat):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        if self._norm:
            norm = torch.pow(A_Mtx.sum(dim=0), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm
        rst = A_Mtx.mm(feat).mm(self.weight)
        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, num_classes)

    def forward(self, graph, inputs):
        h = self.gcn_layer1(graph, inputs.to(self.device)).to(self.device)
        h = F.relu(h)
        h = self.gcn_layer2(graph, h.cuda()).cuda()
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GAT, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False).cuda()
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False).cuda()

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)
        return {'e': src_e}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h):
        z = self.linear_func(h)
        graph.ndata['z'] = z
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata.pop('h')

class SelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=8, attention_probs_dropout_prob=0.2,
                 max_position_embeddings=10, position_embedding_type='absolute'):
        super().__init__()
        if hidden_size % num_attention_heads != 0 and not hasattr( "embedding_size"):
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            """
                relative position embedding can also be used in rumor detection area. For the replyes that far from the 
                source message, it should be assigned smaller attentions, while 
            """
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children): # recursive
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, in_features, out_features):
        '''TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order):
        '''Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return h, c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration # the source node of edge is sorted from small node idx

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consexcutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])

class BU_RvNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(BU_RvNN, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = in_feats
        self.prop_hidden_size = h_feats
        self.gru_cell = nn.GRUCell(self.sent_hidden_size, self.prop_hidden_size)

    def get_op_pairs(self, trees, resort_sizes):
        op_pairs_list = []
        for idx, tree in enumerate(trees):
            current_layer = [tree]
            op_pairs = []
            while current_layer:
                ops = []
                new_layer = []
                for sub_tree in current_layer:
                    if sub_tree.num_children != 0:
                        op = (sub_tree.root_idx + resort_sizes[idx],
                              [child.root_idx + resort_sizes[idx] for child in sub_tree.children]
                              )
                        new_layer.extend(sub_tree.children)
                    else:
                        op = (sub_tree.root_idx + resort_sizes[idx],
                              -1
                              )
                    ops.append(op)
                current_layer = new_layer
                op_pairs.append(ops)
            op_pairs_list.append(op_pairs)
        layers = []
        tree_num = len(op_pairs_list)
        while True:
            current_layer = []
            for i in range(tree_num):
                if op_pairs_list[i]:
                    current_layer.extend(op_pairs_list[i].pop(0))
            if current_layer:
                layers.append(current_layer)
            else:
                break
        return layers

    def forward(self, trees, inputs):
        depths = [tree.depth() for tree in trees]
        sizes = [tree.size() for tree in trees]
        nodes = sum(sizes)
        assert nodes == len(inputs)
        all_hiddens = torch.zeros([nodes, self.prop_hidden_size], device=self.device)
        resort_sizes = [sum(sizes[:i]) for i in range(len(sizes))]
        all_op_pairs = self.get_op_pairs(trees, resort_sizes)
        assert len(all_op_pairs) == max(depths) + 1
        for i in reversed(range(max(depths))):
            ch_hidden_list = []
            ipt_list = []
            for op_pair in all_op_pairs[i]:
                if op_pair[1] == -1:
                    ch_hidden = torch.zeros([self.prop_hidden_size], device=self.device)
                else:
                    ch_hidden = all_hiddens[op_pair[-1], :].mean(dim=0)
                ch_hidden_list.append(ch_hidden)
                ipt_list.append(op_pair[0])
            input_tensors = inputs[ipt_list, :]
            hidden_tensors = torch.stack(ch_hidden_list)
            new_hiddens = self.gru_cell(input_tensors, hidden_tensors)
            all_hiddens[ipt_list, :] = new_hiddens
        root_idxs = [op_pair[0] for op_pair in all_op_pairs[0]]
        return all_hiddens[root_idxs]

class TD_RvNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(TD_RvNN, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = in_feats
        self.prop_hidden_size = h_feats
        self.gru_cell = nn.GRUCell(self.sent_hidden_size, self.prop_hidden_size)

    def forward(self, trees, inputs):
        depths = [tree.depth() for tree in trees]
        sizes = [tree.size() for tree in trees]
        nodes = sum(sizes)
        assert nodes == len(inputs)
        hiddens = torch.zeros([nodes, self.prop_hidden_size], device=self.device)
        resort_sizes = [sum(sizes[:i]) for i in range(len(sizes))]
        leaf_nodes_lists = [[idx+resort_sizes[i] for idx in tree.leaf_node_idxs()]
                          for i, tree in enumerate(trees)]
        for i in range(max(depths)):
            if i == 0:
                input_idxs = [tree.root_idx + resort_sizes[i] for i, tree in
                              enumerate(trees)]  # every tree start from 0
                input_tensors = inputs[input_idxs, :]
                hidden_tensors = torch.zeros([len(input_idxs), self.prop_hidden_size], device=self.device)
                parent_idxs, new_trees, resort_sizes = self.get_next_layer(trees, resort_sizes)
            else:
                hidden_tensors = hiddens[parent_idxs, :]
                input_idxs = [tree.root_idx + resort_sizes[i]
                              for i, tree in enumerate(new_trees)]  # every tree start from 0
                input_tensors = inputs[input_idxs, :]
                parent_idxs, new_trees, resort_sizes = self.get_next_layer(new_trees, resort_sizes)
            hiddens[input_idxs, :] = self.gru_cell(input_tensors, hidden_tensors)
        leaf_hiddens = [hiddens[leaf_nodes, :].max(dim=0)[0] for leaf_nodes in leaf_nodes_lists]
        return torch.stack(leaf_hiddens)

def obtainSubTrees(tree):
    subtrees = []
    childrens = tree.children
    subtrees.append([tree.root_idx] + [chd.root_idx for chd in childrens])
    while len(childrens) != 0 :
        ch_len = len(childrens)
        for ch_tree in childrens:
            if len(ch_tree.children) != 0:
                subtrees.append([ch_tree.root_idx] +
                                [chd.root_idx for chd in ch_tree.children])
                childrens = childrens + ch_tree.children
        childrens = childrens[ch_len:]
    return subtrees

def get_mask(ids, max_seq_len, device=torch.device("cuda:0")):
    masks = torch.zeros(max_seq_len, device=device)
    masks[ids] = 1
    return masks

def collate_batch(items):
    idxs = [item[0] for item in items]
    masks = torch.stack([item[1] for item in items])
    return idxs, (1.0 - masks[:, None, None, :])*-1e10

def get_ops(sub_tree_list, max_seq_len):
    max_depth = max([
        len(sub_trees) for sub_trees in sub_tree_list
    ])
    for i in range(max_depth):
        items = [(idx, get_mask(sub_trees[i], max_seq_len))
                for idx, sub_trees in enumerate(sub_tree_list) if len(sub_trees) > (i)]
        yield collate_batch(items)

class TD_Transformer(nn.Module):
    def __init__(self, config_file):
        super(TD_Transformer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config = BertConfig.from_pretrained(config_file)
        self.transformer = BertEncoder(config)
        self.attention_mu = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, trees, seq_sent_vecs):
        sub_tree_list = [obtainSubTrees(tree) for tree in trees]
        pad_vecs = torch.nn.utils.rnn.pad_sequence(seq_sent_vecs, batch_first=True)
        pad_masks = pad_vecs.data.__eq__(0.0).int().sum(dim=-1).__eq__(pad_vecs.size(1)).float()*(-1e10)
        for (idxs, extend_masks) in get_ops(sub_tree_list, max_seq_len=pad_vecs.size(1)):
            rst = self.transformer(pad_vecs[idxs], attention_mask=extend_masks)
            left_mask = extend_masks.squeeze(1).squeeze(1).unsqueeze(-1) / -1e10
            right_mask = extend_masks.squeeze(1).squeeze(1).unsqueeze(-1) / 1e10 + 1
            pad_vecs[idxs] = left_mask * pad_vecs[idxs] + right_mask * rst.last_hidden_state
        # pad_vecs : [batch, seq, sent_dim]
        alpha = self.attention_mu(pad_vecs).squeeze(-1) + pad_masks
        seq_vecs = (alpha.softmax(dim=1).unsqueeze(-1) * pad_vecs).sum(dim=1)
        return seq_vecs

class BU_Transformer(nn.Module):
    def __init__(self, config_file):
        super(BU_Transformer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config = BertConfig.from_pretrained(config_file)
        self.transformer = BertEncoder(config)
        self.attention_mu = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, trees, seq_sent_vecs):
        sub_tree_list = [reversed(obtainSubTrees(tree)) for tree in trees]
        pad_vecs = torch.nn.utils.rnn.pad_sequence(seq_sent_vecs, batch_first=True)
        pad_masks = pad_vecs.data.__eq__(0.0).int().sum(dim=-1).__eq__(pad_vecs.size(1)).float() * (-1e10)
        for (idxs, extend_masks) in get_ops(sub_tree_list, max_seq_len=pad_vecs.size(1)):
            rst = self.transformer(pad_vecs[idxs], attention_mask=extend_masks)
            left_mask = extend_masks.squeeze(1).squeeze(1).unsqueeze(-1) / -1e10
            right_mask = extend_masks.squeeze(1).squeeze(1).unsqueeze(-1) / 1e10 + 1
            pad_vecs[idxs] = left_mask * pad_vecs[idxs] + right_mask * rst.last_hidden_state
        alpha = self.attention_mu(pad_vecs).squeeze(-1) + pad_masks
        seq_vecs = (alpha.softmax(dim=1).unsqueeze(-1) * pad_vecs).sum(dim=1)
        return seq_vecs

class BiGCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(BiGCN, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = in_feats
        self.prop_hidden_size = h_feats
        self.gcn_layer1_TD = GraphConv(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_TD = GraphConv(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer1_BU = GraphConv(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_BU = GraphConv(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)

    def forward(self, TD_graphs, BU_graphs, inputs):
        num_nodes = [g.num_nodes() for g in TD_graphs]
        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        big_g_TD = dgl.batch(TD_graphs)

        h_TD = self.gcn_layer1_TD(big_g_TD, inputs)
        h_TD = F.relu(h_TD)
        H_TD_inputs = torch.cat([torch.cat([h_TD[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_TD = self.gcn_layer2_TD(big_g_TD, H_TD_inputs)
        H_TD = F.relu(H_TD)
        H_TD_out = torch.cat([torch.cat([H_TD[r_idx:r_idx + num_nodes[i]],
                                         h_TD[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])

        H_mean_TD = torch.stack([
            H_TD_out[r_idx:r_idx + num_nodes[i]].mean(dim=0)
            for i, r_idx in enumerate(root_idxs)
        ])

        big_g_BU = dgl.batch(BU_graphs)
        h_BU = self.gcn_layer1_BU(big_g_BU, inputs)
        h_BU = F.relu(h_BU)
        H_BU_inputs = torch.cat([torch.cat([h_BU[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_BU = self.gcn_layer2_BU(big_g_BU, H_BU_inputs)
        H_BU = F.relu(H_BU)
        H_BU_out = torch.cat([torch.cat([H_BU[r_idx:r_idx + num_nodes[i]],
                                         h_BU[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])

        H_mean_BU = torch.stack([
            H_BU_out[r_idx:r_idx + num_nodes[i]].mean(dim=0)
            for i, r_idx in enumerate(root_idxs)
        ])

        return torch.cat([H_mean_TD, H_mean_BU], dim=1)


class BiGCNV2(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(BiGCNV2, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = in_feats
        self.prop_hidden_size = h_feats
        self.gcn_layer1_TD = GraphConvV2(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_TD = GraphConvV2(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer1_BU = GraphConvV2(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_BU = GraphConvV2(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)

    def forward(self, num_nodes, A_TD:torch.Tensor, A_BU:torch.Tensor, inputs):

        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        h_TD = self.gcn_layer1_TD(A_TD, inputs)
        h_TD = F.relu(h_TD)
        H_TD_inputs = torch.cat([torch.cat([h_TD[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_TD = self.gcn_layer2_TD(A_TD, H_TD_inputs)
        H_TD = F.relu(H_TD)
        H_TD_out = torch.cat([torch.cat([H_TD[r_idx:r_idx + num_nodes[i]],
                                         h_TD[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])

        H_mean_TD = torch.stack([
            H_TD_out[r_idx:r_idx + num_nodes[i]].mean(dim=0)
            for i, r_idx in enumerate(root_idxs)
        ])
        h_BU = self.gcn_layer1_BU(A_BU, inputs)
        h_BU = F.relu(h_BU)
        H_BU_inputs = torch.cat([torch.cat([h_BU[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_BU = self.gcn_layer2_BU(A_BU, H_BU_inputs)
        H_BU = F.relu(H_BU)
        H_BU_out = torch.cat([torch.cat([H_BU[r_idx:r_idx + num_nodes[i]],
                                         h_BU[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])

        H_mean_BU = torch.stack([
            H_BU_out[r_idx:r_idx + num_nodes[i]].mean(dim=0)
            for i, r_idx in enumerate(root_idxs)
        ])

        return torch.cat([H_mean_TD, H_mean_BU], dim=1)

class BiGCNV3(BiGCNV2):
    def __init__(self, in_feats, h_feats, num_attention_heads=8):
        super(BiGCNV2, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sent_hidden_size = in_feats
        self.prop_hidden_size = h_feats
        self.gcn_layer1_TD = GraphConvV2(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_TD = GraphConvV2(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer1_BU = GraphConvV2(in_feats, h_feats).to(self.device, non_blocking=True)
        self.gcn_layer2_BU = GraphConvV2(h_feats+in_feats, h_feats).to(self.device, non_blocking=True)
        if self.prop_hidden_size % num_attention_heads != 0 and not hasattr( "embedding_size"):
            raise ValueError(
                f"The hidden size ({self.prop_hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.prop_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.ln_TD = nn.LayerNorm(self.prop_hidden_size, eps=1e-8)
        self.ln_BU = nn.LayerNorm(self.prop_hidden_size, eps=1e-8)

        self.query_TD = nn.Linear(self.prop_hidden_size, self.all_head_size)
        self.key_TD = nn.Linear(self.prop_hidden_size, self.all_head_size)
        self.value_TD = nn.Linear(self.prop_hidden_size, self.all_head_size)

        self.query_BU = nn.Linear(self.prop_hidden_size, self.all_head_size)
        self.key_BU = nn.Linear(self.prop_hidden_size, self.all_head_size)
        self.value_BU = nn.Linear(self.prop_hidden_size, self.all_head_size)

    def self_attention_TD(self, hidden:torch.Tensor, attn_mask:torch.Tensor):
        assert hidden.dim() == 2 and attn_mask.dim() == 2
        query, key, val = self.query_TD(hidden), self.key_TD(hidden), self.value_TD(hidden)
        m_head_q = query.reshape(query.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)
        m_head_k = key.reshape(key.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        m_head_v = val.reshape(val.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)

        attn_logits = torch.matmul(m_head_q, m_head_k)
        attn_score = (attn_logits + attn_mask).softmax(dim=1)

        context_outs = torch.matmul(attn_score, m_head_v).permute(1, 0, 2).contiguous().reshape(
            attn_mask.size(0), self.all_head_size
        )
        return context_outs

    def self_attention_BU(self, hidden:torch.Tensor, attn_mask:torch.Tensor):
        assert hidden.dim() == 2 and attn_mask.dim() == 2
        query, key, val = self.query_BU(hidden), self.key_BU(hidden), self.value_BU(hidden)
        m_head_q = query.reshape(query.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)
        m_head_k = key.reshape(key.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        m_head_v = val.reshape(val.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)

        attn_logits = torch.matmul(m_head_q, m_head_k)
        attn_score = (attn_logits + attn_mask).softmax(dim=1)

        context_outs = torch.matmul(attn_score, m_head_v).permute(1, 0, 2).contiguous().reshape(
            attn_mask.size(0), self.all_head_size
        )
        return context_outs

    def forward(self, num_nodes:List, A_TD:torch.Tensor, A_BU:torch.Tensor, inputs):
        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        attn_mask = torch.ones(sum(num_nodes), sum(num_nodes), dtype=torch.float32, device=self.device)*(-1e10)
        for root_idx, num_node in zip(root_idxs, num_nodes):
            attn_mask[root_idx:root_idx+num_node, root_idx:root_idx+num_node] = 0.0

        h_TD = self.gcn_layer1_TD(A_TD, inputs)
        h_TD = F.relu(h_TD)
        H_TD_inputs = torch.cat([torch.cat([h_TD[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_TD = self.gcn_layer2_TD(A_TD, H_TD_inputs)
        H_TD = F.relu(H_TD)
        H_TD_out = torch.cat([torch.cat([H_TD[r_idx:r_idx + num_nodes[i]],
                                         h_TD[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])
        H_TD_attn = self.self_attention_TD(
            self.ln_TD(
                F.relu(H_TD_out)
            ),
            attn_mask
        )

        h_BU = self.gcn_layer1_BU(A_BU, inputs)
        h_BU = F.relu(h_BU)
        H_BU_inputs = torch.cat([torch.cat([h_BU[r_idx:r_idx + num_nodes[i]],
                                            inputs[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                                 for i, r_idx in enumerate(root_idxs)])

        H_BU = self.gcn_layer2_BU(A_BU, H_BU_inputs)
        H_BU = F.relu(H_BU)
        H_BU_out = torch.cat([torch.cat([H_BU[r_idx:r_idx + num_nodes[i]],
                                         h_BU[r_idx:r_idx + 1].repeat([num_nodes[i], 1])], dim=1)
                              for i, r_idx in enumerate(root_idxs)])
        H_BU_attn = self.self_attention_BU(
            self.ln_BU(
                F.relu(H_BU_out)
            ),
            attn_mask
        )
        return torch.cat([H_TD_attn[root_idxs], H_BU_attn[root_idxs]], dim=1)