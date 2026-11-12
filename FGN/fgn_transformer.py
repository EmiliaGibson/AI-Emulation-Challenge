#Adapted from https://github.com/google-deepmind/weathernext/blob/main/weathernext

import numpy as np
import scipy as sp
import torch
import dataclasses
import layers
import torch.nn as nn
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import HeteroData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Any, Callable, Literal, Optional, Tuple
from layers import ConditionalLayerNorm


def get_mask_block_size(mask):
  """Get blocksize of the adjacency matrix (attn mask) for the permuted mesh."""
  # sub-diagonal bandwidth
  lbandwidth = (
      np.arange(mask.shape[0]) - (mask != 0).argmax(axis=0) + 1).max()
  # super-diagonal bandwidth
  ubandwidth = (
      (mask.shape[0]-1) - np.argmax(mask[::-1,:] != 0, axis=0
                                    ) - np.arange(mask.shape[0]) + 1).max()
  block_size = np.maximum(lbandwidth, ubandwidth)
  return block_size

# convert adjacency matrix into equally sized blocks  
def mask_block_diags(mask: csr_matrix,
                     num_padding_nodes: int,
                     block_size: int) -> np.ndarray:
  """Pad and reshape mask diag, super-diag and sub-diag blocks."""
  # add zero padding to mask
  mask_padding_rows = csr_matrix(
      (num_padding_nodes, mask.shape[1]), dtype=np.int32)
  mask = sp.sparse.vstack([mask, mask_padding_rows])
  mask_padding_cols = csr_matrix(
      (mask.shape[0], num_padding_nodes), dtype=np.int32)
  mask = sp.sparse.hstack([mask, mask_padding_cols])
  assert (mask.shape[-1] % block_size) == 0
  
  mask_diag_blocks = np.stack(
      [np.array(mask[i * block_size : (i + 1) * block_size,
                      i * block_size : (i + 1) * block_size,
                      ].toarray())
       for i in range(mask.shape[0] // block_size)])
  
  mask_upper_diag_blocks = np.stack(
      [np.array(mask[i * block_size : (i + 1) * block_size,
                      (i + 1) * block_size : (i + 2) * block_size,
                      ].toarray())
       for i in range(mask.shape[0] // block_size - 1)]
      + [np.zeros((block_size, block_size), dtype=mask.dtype)])
  mask_lower_diag_blocks = np.stack(
      [np.zeros((block_size, block_size), dtype=mask.dtype)]
      + [np.array(mask[(i + 1) * block_size : (i + 2) * block_size,
                        i * block_size : (i + 1) * block_size,
                        ].toarray())
         for i in range(mask.shape[0] // block_size - 1)])
  mask = np.stack( 
      [mask_diag_blocks, mask_upper_diag_blocks, mask_lower_diag_blocks]
  )
  mask = np.expand_dims(mask, (0, 3)) 
  return torch.tensor(mask, dtype = torch.bool)  

#Set up model parameters

@dataclasses.dataclass
class _ModelConfig:
  """Transformer config."""
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  num_layers: int
  # Primary width, the number of channels on the carrier path.
  d_model: int
  # Number of heads for self-attention.
  num_heads: int
  # Mask block size.
  mask_block_size: int
  # Number of channels per-head for self-attn QK computation.
  key_size: Optional[int] = None
  # Number of channels per-head for self-attn V computation.
  value_size: Optional[int] = None
  # Activation to use
  activation: str = 'gelu'
  # Init scale for ffw layers (divided by num_layers)
  ffw_winit_mult: float = 2.0
  # Init scale for final ffw layer (divided by depth)
  ffw_winit_final_mult: float = 2.0
  # Init scale for mha proj (divided by depth).
  attn_winit_mult: float = 2.0
  # Init scale for mha w (divided by depth).
  attn_winit_final_mult: float = 2.0
  # Number of hidden units in the MLP blocks. Defaults to 4 * d_model.
  ffw_hidden: Optional[int] = None

  def __post_init__(self):
    if self.ffw_hidden is None:
      self.ffw_hidden = 4 * self.d_model
    # Compute key_size and value_size from d_model // num_heads.
    if self.key_size is None:
      if self.d_model % self.num_heads != 0:
        raise ValueError('num_heads has to divide d_model exactly')
      self.key_size = self.d_model // self.num_heads
    if self.value_size is None:
      if self.d_model % self.num_heads != 0:
        raise ValueError('num_heads has to divide d_model exactly')
      self.value_size = self.d_model // self.num_heads

#construct tri-block-diagram mha
def triblockdiag_softmax(logits: Tuple[torch.tensor, torch.tensor, torch.tensor]
                         ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
  """Softmax given the diag, upper diag, and lower diag logit blocks."""

  logits_d, logits_u, logits_l = logits

  # calculate max logit for numerical stability
  m = torch.amax(torch.stack([
      logits_d.detach().amax(-1, keepdims=True),
      logits_u.detach().amax(-1, keepdims=True),
      logits_l.detach().amax(-1, keepdims=True)]), axis=0)

  unnormalized_d = torch.exp(logits_d - m)
  unnormalized_u = torch.exp(logits_u - m)
  unnormalized_l = torch.exp(logits_l - m)

  denom = (
      unnormalized_d.sum(-1, keepdims=True)
      + unnormalized_u.sum(-1, keepdims=True)
      + unnormalized_l.sum(-1, keepdims=True)
  )

  logits_d = unnormalized_d / denom
  logits_u = unnormalized_u / denom
  logits_l = unnormalized_l / denom

  return (logits_d, logits_u, logits_l)

class MultiHeadLinear(nn.Module):
  def __init__(self, qkv: str, cfg: _ModelConfig):
    super().__init__()
    self.cfg = cfg
    self.head_size = cfg.value_size if qkv == 'v' else cfg.key_size
    self.mha_proj = nn.LazyLinear(cfg.num_heads*self.head_size, bias=False)
  
  def forward(self, x: torch.tensor):
    out = self.mha_proj(x)
    shape = out.shape[:-1] + (self.cfg.num_heads, self.head_size)
    out = out.reshape(shape)
    return out

class TriBlockDiagMHA(nn.Module):
  def __init__(self, cfg: _ModelConfig):
    super().__init__()
    self.cfg = cfg
    self.mha_final = nn.LazyLinear(self.cfg.d_model, bias=False)
    self.mha_proj_q = MultiHeadLinear('q', cfg)  
    self.mha_proj_k = MultiHeadLinear('k', cfg)  
    self.mha_proj_v = MultiHeadLinear('v', cfg)  
  
  def forward(self, q_input: torch.tensor, kv_input: torch.tensor,
                     mask: torch.tensor) -> torch.tensor:
    """Triblockdiag multihead attention."""

    # q_inputs, kv_input: (batch, num_blocks, block_size, num_heads, d_model)
    q = self.mha_proj_q(q_input)
    k = self.mha_proj_k(kv_input)
    v = self.mha_proj_v(kv_input)
    
    k = nn.functional.pad(k, (0, 0, 0, 0, 0, 0, 1, 1, 0, 0)) #pad dim =1
    v = nn.functional.pad(v, (0, 0, 0, 0, 0, 0, 1, 1, 0, 0)) #pad dim =1
    def qk_prod(queries, keys):
      return torch.einsum('bnqhd,bnkhd->bnhqk', queries, keys)

    # q shape is (batch, num_blocks, block_size, num_heads, qk_dim)
    # k shape is (batch, num_blocks + 2, block_size, num_heads, qk_dim)
    logits_d = qk_prod(q, k[:, 1:-1, ...]) * self.cfg.key_size**-0.5 
    logits_u = qk_prod(q, k[:, 2:, ...]) * self.cfg.key_size**-0.5 
    logits_l = qk_prod(q, k[:, :-2, ...]) * self.cfg.key_size**-0.5  

    # apply mask #if using torch.float32
    logits_d = torch.where(mask[:, 0, ...], logits_d, -1e30)
    logits_u = torch.where(mask[:, 1, ...], logits_u, -1e30)
    logits_l = torch.where(mask[:, 2, ...], logits_l, -1e30)

    logits_d, logits_u, logits_l = triblockdiag_softmax((logits_d, logits_u, logits_l))
    
    def av_prod(attn_weights, values):
      return torch.einsum('bnhqk,bnkhd->bnqhd', attn_weights, values)

    out_d = av_prod(logits_d, v[:, 1:-1, ...])
    out_u = av_prod(logits_u, v[:, 2:, ...])
    out_l = av_prod(logits_l, v[:, :-2, ...])
    # x shape is (batch, num_blocks, block_size, num_heads, d_model)
    x = out_d + out_u + out_l

    x = torch.reshape(x, x.shape[:-2] + (self.cfg.num_heads * self.cfg.value_size,)) 
  
    
    # attn_winit_final = hk.initializers.VarianceScaling(
    #     self.cfg.attn_winit_final_mult / self.cfg.num_layers)
    # x = hk.Linear(self.cfg.d_model, name='mha_final', w_init=attn_winit_final)(x)
    x = self.mha_final(x)
    return x


class FFW(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.ffw_up = nn.LazyLinear(cfg.ffw_hidden)
    self.gelu = nn.GELU()
    self.ffw_down = nn.LazyLinear(cfg.d_model)    
      
  def forward(self, x: torch.tensor) -> torch.tensor:
    """Feed-forward block.""" 
    x = self.ffw_up(x)
    x = self.gelu(x)
    return self.ffw_down(x)


class Block(nn.Module):
  """Transformer block (mha and ffw for embedded inputs)."""

  def __init__(self, cfg, mask, num_nodes, num_padding_nodes):
    super().__init__()
    self._cfg = cfg
    self.register_buffer("mask", mask)
    self.num_nodes = num_nodes
    self.num_padding_nodes = num_padding_nodes
    self.triblockdiag_mha = TriBlockDiagMHA(cfg)
    self.ffw = FFW(cfg)
    #global norm vector has size 8
    self.norm_conditioning_attn = ConditionalLayerNorm(cfg.d_model,8)
    self.norm_conditioning_ffw = ConditionalLayerNorm(cfg.d_model,8)

  def forward(self, x, global_norm_conditioning=torch.tensor):
    # x shape is (batch, num_nodes, feature_dim)
    def attn(x):
        # We pad -> reshape -> compute attn -> reshape -> select at each block
        # so as to avoid complications involved in making the norm layers and
        # ffw blocks account for the padding. However, this might be decreasing
        # efficiency.

        # Add padding so that number of nodes is divisible into blocks
        x = nn.functional.pad(x, (0, 0, 0, self.num_padding_nodes, 0, 0))
        x = x.reshape(x.shape[0],
                      x.shape[1]//self._cfg.mask_block_size,
                      self._cfg.mask_block_size,
                      x.shape[-1])
        x = self.triblockdiag_mha(x, x, mask=self.mask.detach().clone())
        x = x.reshape(x.shape[0],
                      self.num_nodes + self.num_padding_nodes,
                      x.shape[-1])
        return x[:,:self.num_nodes, :]


    x = x + attn(
        self.norm_conditioning_attn(
            x, global_norm_conditioning)
        )
    x = x + self.ffw(self.norm_conditioning_ffw(x, global_norm_conditioning)
        )

    return x


class Transformer(nn.Module):
  """Main transformer module that processes embeddings.

  All but the very first and very last layer of a 'classic' Transformer:
  Receives already embedded inputs instead of discrete tokens.
  Outputs an embedding for each 'node'/'position' rather than logits.
  """

  def __init__(self,
               adj_mat: sp.sparse.csr_matrix,
               attention_k_hop: int,
               num_heads=1,
               **kwargs):
    super().__init__()

    # Construct mask and deduce block size.
    mask = adj_mat ** attention_k_hop
    mask_block_size = get_mask_block_size(mask)
    # logging.info('mask_block_size: %s.', mask_block_size)
    self.num_nodes = adj_mat.shape[0]
    
    # we will stack the nodes in blocks of 'block_size' nodes, so we need to
    # pad the input such that (num_nodes + num_padding_nodes) % block_size = 0
    self.num_padding_nodes = int(np.ceil(
        mask.shape[0]/mask_block_size)*mask_block_size
                                   - mask.shape[0])
    self.mask = mask_block_diags(
        mask, self.num_padding_nodes, mask_block_size)
    
      
    # Construct config for use within class.
    self._cfg = _ModelConfig(
        mask_block_size=mask_block_size,
        num_heads=num_heads,
        **kwargs,
    )

    self.blocks = nn.ModuleList([Block(self._cfg,self.mask,self.num_nodes,self.num_padding_nodes ) for i in range(self._cfg.num_layers)])
    self.final_norm_conditioning = ConditionalLayerNorm(self._cfg.d_model,8)
      
  def forward(self, node_features, global_norm_conditioning: torch.tensor):
    # node_features expected to have shape (batch, num_nodes, d)
    x = node_features
    for i_block in range(self._cfg.num_layers):
      x = self.blocks[i_block](x, global_norm_conditioning)
      

    x = self.final_norm_conditioning(x, global_norm_conditioning)

    return x

# construct gridtomesh encoder/decoder
class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        ## NB Sequential does not work in conjuction with DataParallel or DDP
        # self.layer_seq = Sequential('x, edge_index', [
        #     (GCNConv(-1, hidden), 'x, edge_index -> x'),
        #     nn.SiLU(inplace=False),
        #     (GCNConv(hidden,out), 'x, edge_index -> x'),
        #     (nn.LayerNorm(hidden)),
        # ])
        self.lin1 = nn.LazyLinear(hidden)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(hidden,hidden)
        self.norm = nn.LayerNorm(hidden)

    
    def forward(self, x):
        h = self.lin1(x)
        h = self.gelu(h)
        h = self.lin2(h)
        out = self.norm(h)
        return out


class DecoderMLP(nn.Module):
    def __init__(self,hidden,out):
        super().__init__()
        if out is None:
            out = hidden
        # self.layer_seq = Sequential('x, edge_index', [
        #     (GCNConv(-1, hidden), 'x, edge_index -> x'),
        #     nn.SiLU(inplace=False),
        #     (GCNConv(hidden,out), 'x, edge_index -> x'),
        # ])
        self.lin1 = nn.Linear(hidden, hidden)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(hidden,out)

        
    
    
    def forward(self, x):
        h = self.lin1(x)
        h = self.gelu(h)
        h = self.lin2(h)
        return h


class FeatureEmbedding(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        #node embeddings
        self.grid_node_embedding = MLP(hidden)
        self.mesh_node_embedding = MLP(hidden)
        #edge embeddings
        self.G2M_embedding = MLP(32)
        self.mesh_embedding = MLP(hidden)
        self.M2G_embedding = MLP(hidden)
    
    
    def forward(self,attrs,X_feats):      
        '''
        X_feats: tensor (B, nodes, feats)
        '''
        mesh_node_feats, mesh_edge_attrs, G2M_edge_attrs, M2G_edge_attrs = attrs
        
        embedded_mesh_nodes = self.mesh_node_embedding(mesh_node_feats.clone()) #(nodes, feats)
        embedded_grid_nodes = self.grid_node_embedding(X_feats.clone()) #(B, nodes, feats)
                                                        
        embedded_mesh_attr = self.mesh_embedding(mesh_edge_attrs.clone()) #(edges, feats)
        embedded_G2M_attr = self.G2M_embedding(G2M_edge_attrs.clone()) #(edges, feats)
        embedded_M2G_attr = self.M2G_embedding(M2G_edge_attrs.clone()) #(edges, feats)

        attrs = {'grid_nodes': embedded_grid_nodes,
                 'mesh_nodes': embedded_mesh_nodes.unsqueeze(0),
                 ('grid_nodes', 'G2M_edges', 'mesh_nodes') : embedded_G2M_attr.unsqueeze(0),
                 ('mesh_nodes', 'mesh_edges', 'mesh_nodes'): embedded_mesh_attr.unsqueeze(0),
                 ('mesh_nodes', 'M2G_edges', 'grid_nodes') : embedded_M2G_attr.unsqueeze(0)}

        return attrs



class INLayer(nn.Module):
    def __init__(self, aggr_fn,hidden1, hidden2):
        super().__init__()
        self.aggr_fn = aggr_fn

        self.rm = MLP(hidden1)
        self.om = MLP(hidden2)

    def aggregate(self, attrs, num_nodes,nodes, edges, edge_updates, receiver_ind):        
        aggr_ = self.aggr_fn(edge_updates, receiver_ind, dim_size = num_nodes[edges[-1]])        
        return torch.cat((attrs[nodes].expand(len(aggr_), -1, -1),aggr_),dim=2)
    
    def forward(self, attrs, edge_indices, num_nodes, nodes, edges):
        sender = attrs[edges[0]][:, edge_indices[edges][0]]
        receiver = attrs[edges[2]][:,edge_indices[edges][1]]

        martial =  torch.cat(
                    (sender, receiver.expand(len(sender), -1, -1),
                     attrs[edges].expand(len(sender), -1, -1)),dim=2,
                     ).to(torch.float32)
        edge_update = self.rm(martial)
        aggr = self.aggregate(attrs,num_nodes, nodes, edges, edge_update, edge_indices[edges][1])
        node_update = self.om(aggr)
        
        return edge_update, node_update


class G2MEncoder(nn.Module):
    def __init__(self, aggr_fn ,hidden ):
        super().__init__()
        self.aggr_fn = aggr_fn
        self.hidden=hidden
        self.in_layer= INLayer(self.aggr_fn, 32, self.hidden)
        self.grid_nodes_mlp = MLP(self.hidden)
    
    def forward(self, attrs, edge_indices, num_nodes):
        edge_update, mesh_node_update = self.in_layer(attrs, edge_indices, num_nodes, 'mesh_nodes', ('grid_nodes', 'G2M_edges', 'mesh_nodes'))
        grid_node_update = self.grid_nodes_mlp(attrs['grid_nodes'])
        attrs['mesh_nodes'] = attrs['mesh_nodes'] + mesh_node_update
        attrs[('grid_nodes', 'G2M_edges', 'mesh_nodes')] =  attrs[('grid_nodes', 'G2M_edges', 'mesh_nodes')] + edge_update
        attrs['grid_nodes'] = attrs['grid_nodes'] + grid_node_update

        return attrs


class M2GDecoder(nn.Module):
    def __init__(self, aggr_fn, hidden ):
        super().__init__()
        self.aggr_fn = aggr_fn
        self.hidden=hidden
        self.in_layer= INLayer(self.aggr_fn,self.hidden, self.hidden)
        self.grid_nodes_mlp = DecoderMLP(self.hidden, out=7)
        
    def forward(self, attrs, edge_indices, num_nodes):
        edge_update, grid_node_update = self.in_layer(attrs, edge_indices, num_nodes, 'grid_nodes', ('mesh_nodes', 'M2G_edges', 'grid_nodes'))
        
        #Force a .clone() on the grid nodes and their update to break the version tracking chain
        updated_grid_nodes = attrs['grid_nodes'] + grid_node_update        
        attrs['grid_nodes'] = updated_grid_nodes.clone()

        return self.grid_nodes_mlp(attrs['grid_nodes'])


class FGN(nn.Module):
    def __init__(self, graph, num_layers = 4,num_heads=4, attention_k_hop = 4, aggr_fn = SumAggregation(), hidden=512, **kwargs):
        super().__init__()
        #register buffers for model sending to device ddp
        self.register_buffer('num_grid_nodes', torch.tensor([graph['grid_nodes'].num_nodes]),persistent=False)
        self.register_buffer('num_mesh_nodes', torch.tensor([graph['mesh_nodes'].num_nodes]),persistent=False)
        self.register_buffer('mesh_node_feats', graph['mesh_nodes'].x,persistent=False)
        self.register_buffer('mesh_edge_attrs', graph['mesh_nodes', 'mesh_edges', 'mesh_nodes'].edge_attr,persistent=False)
        self.register_buffer('G2M_edge_attrs', graph['grid_nodes', 'G2M_edges', 'mesh_nodes'].edge_attr,persistent=False)
        self.register_buffer('M2G_edge_attrs', graph['mesh_nodes', 'M2G_edges', 'grid_nodes'].edge_attr,persistent=False)      
        self.register_buffer('mesh_edge_indices',graph['mesh_nodes', 'mesh_edges', 'mesh_nodes'].edge_index,persistent=False)
        self.register_buffer('G2M_edge_indices', graph['grid_nodes', 'G2M_edges', 'mesh_nodes'].edge_index,persistent=False)
        self.register_buffer('M2G_edge_indices', graph['mesh_nodes', 'M2G_edges', 'grid_nodes'].edge_index,persistent=False)
                
        # # get mesh edges from which to construct adjacency matrix
        mesh_edge_index = graph['mesh_nodes', 'mesh_edges', 'mesh_nodes'].edge_index
        #adjacency matrix
        adj = csr_matrix((np.ones(graph['mesh_nodes', 'mesh_edges', 'mesh_nodes'].num_edges), (mesh_edge_index[0].numpy(),mesh_edge_index[1].numpy())))
        perm = reverse_cuthill_mckee(adj,symmetric_mode=True)
        self.adj_permuted = (adj)[perm, :][:, perm]
        self.perm = torch.tensor(perm.copy())
        self.aggr_fn = aggr_fn
        self.hidden=hidden
        self.num_layers=num_layers
        self.num_heads = num_heads
        self.embed_mlp =  FeatureEmbedding(self.hidden)
                
        self.g2m_encoder = G2MEncoder(self.aggr_fn, self.hidden)
        self.graph_transformer = Transformer(self.adj_permuted, attention_k_hop, num_heads, d_model= hidden, num_layers = num_layers, **kwargs)
        self.m2g_decoder = M2GDecoder(self.aggr_fn, self.hidden)

        self.attrs = (self.mesh_node_feats, self.mesh_edge_attrs, self.G2M_edge_attrs, self.M2G_edge_attrs)
        self.edge_indices = {('mesh_nodes', 'mesh_edges', 'mesh_nodes') : self.mesh_edge_indices, 
                   ('grid_nodes', 'G2M_edges', 'mesh_nodes') : self.G2M_edge_indices, 
                   ('mesh_nodes', 'M2G_edges', 'grid_nodes') : self.M2G_edge_indices
                            }
        self.num_nodes = {'mesh_nodes': self.num_mesh_nodes, 
                          'grid_nodes': self.num_grid_nodes}
    
    def update_dict_buffers(self):
        self.attrs = (self.mesh_node_feats, self.mesh_edge_attrs, self.G2M_edge_attrs, self.M2G_edge_attrs)
        
        
        self.edge_indices = {('mesh_nodes', 'mesh_edges', 'mesh_nodes') : self.mesh_edge_indices.clone(), 
                   ('grid_nodes', 'G2M_edges', 'mesh_nodes') : self.G2M_edge_indices.clone(), 
                   ('mesh_nodes', 'M2G_edges', 'grid_nodes') : self.M2G_edge_indices.clone()
                            }
        self.num_nodes = {'mesh_nodes': self.num_mesh_nodes, 
                          'grid_nodes': self.num_grid_nodes}
        self.graph_transformer
        

    def forward(self, X_feats, global_norm_conditioning):
        #batch learning
        
        
        embedded_attrs = self.embed_mlp(self.attrs ,X_feats)
        embedded_attrs = self.g2m_encoder(embedded_attrs, self.edge_indices, self.num_nodes)
        

        embedded_attrs['mesh_nodes'] = self.graph_transformer(embedded_attrs['mesh_nodes'], global_norm_conditioning)[:,self.perm,:]
        node_updates =  self.m2g_decoder(embedded_attrs, self.edge_indices, self.num_nodes)
        return node_updates
