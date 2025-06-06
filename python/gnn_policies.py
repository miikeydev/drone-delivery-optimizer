"""
Custom policies with GNN feature extractors
"""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from gnn_extractor import LocalK10GNN, LocalK10GNNSimple


class GNNPolicy(ActorCriticPolicy):
    """
    PPO Policy with GNN feature extractor
    """
    def __init__(self, *args, **kwargs):
        # Extract GNN-specific kwargs
        gnn_kwargs = {}
        if 'gnn_hidden_dim' in kwargs:
            gnn_kwargs['hidden_dim'] = kwargs.pop('gnn_hidden_dim')
        if 'gnn_heads' in kwargs:
            gnn_kwargs['heads'] = kwargs.pop('gnn_heads')
        if 'gnn_layers' in kwargs:
            gnn_kwargs['gnn_layers'] = kwargs.pop('gnn_layers')
        if 'gnn_aggregation' in kwargs:
            gnn_kwargs['aggregation'] = kwargs.pop('gnn_aggregation')
        if 'gnn_features_dim' in kwargs:
            gnn_kwargs['features_dim'] = kwargs.pop('gnn_features_dim')
        
        super().__init__(
            *args, 
            **kwargs,
            features_extractor_class=LocalK10GNN,
            features_extractor_kwargs=gnn_kwargs
        )


class MaskableGNNPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO Policy with GNN feature extractor
    """
    def __init__(self, *args, **kwargs):
        # Extract GNN-specific kwargs
        gnn_kwargs = {}
        if 'gnn_hidden_dim' in kwargs:
            gnn_kwargs['hidden_dim'] = kwargs.pop('gnn_hidden_dim')
        if 'gnn_heads' in kwargs:
            gnn_kwargs['heads'] = kwargs.pop('gnn_heads')
        if 'gnn_layers' in kwargs:
            gnn_kwargs['gnn_layers'] = kwargs.pop('gnn_layers')
        if 'gnn_aggregation' in kwargs:
            gnn_kwargs['aggregation'] = kwargs.pop('gnn_aggregation')
        if 'gnn_features_dim' in kwargs:
            gnn_kwargs['features_dim'] = kwargs.pop('gnn_features_dim')
        
        super().__init__(
            *args, 
            **kwargs,
            features_extractor_class=LocalK10GNN,
            features_extractor_kwargs=gnn_kwargs
        )


class SimpleGNNPolicy(ActorCriticPolicy):
    """
    PPO Policy with simplified GNN extractor (faster training)
    """
    def __init__(self, *args, **kwargs):
        gnn_kwargs = {}
        if 'gnn_hidden_dim' in kwargs:
            gnn_kwargs['hidden_dim'] = kwargs.pop('gnn_hidden_dim')
        if 'gnn_heads' in kwargs:
            gnn_kwargs['heads'] = kwargs.pop('gnn_heads')
        if 'gnn_features_dim' in kwargs:
            gnn_kwargs['features_dim'] = kwargs.pop('gnn_features_dim')
        
        # REMOVED: gnn_layers and gnn_aggregation (not supported by LocalK10GNNSimple)
        if 'gnn_layers' in kwargs:
            kwargs.pop('gnn_layers')  # Remove but don't pass to simple GNN
        if 'gnn_aggregation' in kwargs:
            kwargs.pop('gnn_aggregation')  # Remove but don't pass to simple GNN
        
        super().__init__(
            *args, 
            **kwargs,
            features_extractor_class=LocalK10GNNSimple,
            features_extractor_kwargs=gnn_kwargs
        )


class MaskableSimpleGNNPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO Policy with simplified GNN extractor
    """
    def __init__(self, *args, **kwargs):
        gnn_kwargs = {}
        if 'gnn_hidden_dim' in kwargs:
            gnn_kwargs['hidden_dim'] = kwargs.pop('gnn_hidden_dim')
        if 'gnn_heads' in kwargs:
            gnn_kwargs['heads'] = kwargs.pop('gnn_heads')
        if 'gnn_features_dim' in kwargs:
            gnn_kwargs['features_dim'] = kwargs.pop('gnn_features_dim')
        
        # REMOVED: gnn_layers and gnn_aggregation (not supported by LocalK10GNNSimple)
        if 'gnn_layers' in kwargs:
            kwargs.pop('gnn_layers')  # Remove but don't pass to simple GNN
        if 'gnn_aggregation' in kwargs:
            kwargs.pop('gnn_aggregation')  # Remove but don't pass to simple GNN
        
        super().__init__(
            *args, 
            **kwargs,
            features_extractor_class=LocalK10GNNSimple,
            features_extractor_kwargs=gnn_kwargs
        )
