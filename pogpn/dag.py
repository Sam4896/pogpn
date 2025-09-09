from typing import List, Set, Union, Sequence, Optional, Dict
import os

import networkx as nx
from matplotlib import pyplot as plt
import logging

from gpytorch.mlls import MarginalLogLikelihood
from botorch.models.approximate_gp import SingleTaskVariationalGP
from dataclasses import dataclass, field
from torch import Tensor
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.approximate_gp import InducingPointAllocator

logger = logging.getLogger("DAG SETUP")


@dataclass
class DAGNode:
    """Node in a DAG to be used for POGPN.

    Args:
        name: Name of the node.
        parents: List of parent nodes.
        node_output_dim: Dimension of the node output.
        node_observation: [Optional during initialization] Observation of the node. Later set by the POGPNBase class.

    """

    name: str
    parents: List["DAGNode"]
    node_output_dim: int
    node_observation: Optional[Tensor] = field(default=None, init=False)

    def __post_init__(self):  # noqa: D105
        self.is_root_input = len(self.parents) == 0

    def __repr__(self):
        """Print string representation of the DAGNode."""
        # Extract parent names once, avoiding conditionals in f-string
        parent_names = []
        for p in self.parents:
            parent_names.append(p.name if isinstance(p, DAGNode) else p)

        return (
            f"DAGNode(name={self.name}, parents={parent_names}, "
            f"node_output_dim={self.node_output_dim}, "
            f"is_root_input={self.is_root_input}"
        )

    def calculate_parent_dims(self) -> int:
        """Calculate input dimensions from parent nodes."""
        # More efficient approach without conditionals in the loop
        dims = 0
        for parent in self.parents:
            if not isinstance(parent, DAGNode):
                raise ValueError(f"Parent '{parent}' is not a DAGNode instance.")
            dims += parent.node_output_dim
        return dims


@dataclass
class RootNode(DAGNode):
    """Root node in a DAG to be used for POGPN."""

    def __post_init__(self):  # noqa: D105
        super().__post_init__()
        if not self.is_root_input:
            raise ValueError("RootNode must be a root input node.")


@dataclass
class RegressionNode(DAGNode):
    """Regression node in a DAG to be used for POGPN.

    node_observation_noise: [Optional during initialization] Noise level of the node.
    node_transform: [Optional during initialization] Transform of the node.
    inducing_point_allocator: [Optional during initialization] Inducing point allocator of the node.
    learn_inducing_points: [Optional during initialization] Whether to learn the inducing points.
    node_mll_loss_history: [Optional during initialization] Loss history of the node.
    node_model: [Optional during initialization] Model of the node.
    node_mll: [Optional during initialization] Marginal log likelihood of the node.
    node_sampler: [Optional during initialization] QMC sampler of the node. Used for the MC samples during pathwise training.
    """

    node_model: Optional[SingleTaskVariationalGP] = field(default=None, init=False)
    node_mll: Optional[MarginalLogLikelihood] = field(default=None, init=False)
    node_sampler: Optional[SobolQMCNormalSampler] = field(default=None, init=False)
    node_mll_loss_history: Dict[str, List[float]] = field(
        default_factory=dict, init=False
    )
    node_observation_noise: Optional[float] = field(default=None)
    node_transform: Optional[OutcomeTransform] = field(default=None)
    inducing_point_allocator: Optional[InducingPointAllocator] = field(default=None)
    learn_inducing_points: bool = field(default=True)

    def __post_init__(self):  # noqa: D105
        self.node_mll_loss_history = {
            "log_likelihood": [],
            "kl_divergence": [],
            "log_prior": [],
            "added_loss": [],
        }


@dataclass
class ClassificationNode(DAGNode):
    """Classification node in a DAG to be used for POGPN.

    Can be use in case the intermediate nodes are categorical.
    """

    node_model: Optional[SingleTaskVariationalGP] = field(default=None, init=False)
    node_mll: Optional[MarginalLogLikelihood] = field(default=None, init=False)
    node_sampler: Optional[SobolQMCNormalSampler] = field(default=None, init=False)
    node_mll_loss_history: Dict[str, List[float]] = field(
        default_factory=dict, init=False
    )
    learn_inducing_points: bool = field(default=True)
    inducing_point_allocator: Optional[InducingPointAllocator] = field(default=None)

    def __post_init__(self):  # noqa: D105
        self.node_mll_loss_history = {
            "log_likelihood": [],
            "kl_divergence": [],
            "log_prior": [],
            "added_loss": [],
        }


class DAG(nx.DiGraph):
    """Class for managing the structure of the Directed Acyclic Graph (DAG)."""

    def __init__(self, dag_nodes: Sequence[DAGNode]):
        """Initialize the DAG handler.

        Args:
            dag_nodes: Sequence of nodes. Each node must have:
                - name: str attribute for node identification
                - parents: Sequence of parent names or parent nodes
        """
        super().__init__()

        """Build the DAG based on the dag_nodes' parent relationships."""
        for dag_node in dag_nodes:
            self.add_node(dag_node.name, data=dag_node)
            for parent in dag_node.parents:
                # If parent is a node instance, use its name; otherwise assume it's a name
                parent_name = parent.name if isinstance(parent, DAGNode) else parent
                if parent_name in self:
                    self.add_edge(parent_name, dag_node.name)
                else:
                    raise ValueError(
                        f"Parent '{parent_name}' for '{dag_node.name}' not found in added nodes."
                    )

    def get_node_parents(self, node_name: str) -> List[str]:
        """Get the parents of a given node."""
        if node_name not in self:
            raise ValueError(f"Node '{node_name}' does not exist in the DAG.")
        return get_deterministic_order(list(self.predecessors(node_name)))

    @property
    def root_nodes(self) -> List[str]:
        """Get the root nodes of the DAG."""
        return get_deterministic_order(
            [node for node, degree in self.in_degree() if degree == 0]
        )

    def nodes_without_root_nodes(self, nodes) -> List[str]:
        """Get the subset of nodes from the provided list without the root nodes."""
        return self.get_deterministic_topological_sort_subset(
            set(nodes) - set(self.root_nodes)
        )

    def get_deep_nodes(self, nodes: List[str]) -> List[str]:
        """Get the deep nodes of the DAG."""
        deep_nodes = []
        for node_name in nodes:
            node_parents = self.get_node_parents(node_name)
            has_only_parent_root_nodes = all(
                parent in self.root_nodes for parent in node_parents
            )

            if not has_only_parent_root_nodes:
                deep_nodes.append(node_name)
        return deep_nodes

    def get_full_deterministic_topological_sort(self) -> List[str]:
        """Get the deterministic topological sort of the DAG."""
        if not hasattr(self, "_cached_full_topo_sort"):
            # Use nx.topological_sort directly and sort each generation
            sorted_generations = list(nx.topological_generations(self))
            self._cached_full_topo_sort = [
                node
                for generation in sorted_generations
                for node in get_deterministic_order(generation)
            ]
        return self._cached_full_topo_sort

    def get_ancestors_deterministic_topological_order(
        self, node_name: str
    ) -> List[str]:
        """Get ancestors of a node in topological generational order where each generation is deterministic sorted."""
        if node_name not in self:
            raise ValueError(f"Node '{node_name}' does not exist in the DAG.")

        # Get all ancestors
        ancestors = nx.ancestors(self, node_name)

        # Get the full topological sort
        full_topo = self.get_full_deterministic_topological_sort()

        # Filter to only include ancestors and sort them according to the full topological order
        return [node for node in full_topo if node in ancestors]

    def get_deterministic_topological_sort_subset(
        self, nodes_subset: Union[List[str], Set[str]]
    ) -> List[str]:
        """Return the nodes in `nodes_subset` sorted topologically according to the full DAG order."""
        full_topo = self.get_full_deterministic_topological_sort()
        if isinstance(nodes_subset, List):
            nodes_subset = set(nodes_subset)
        return [node for node in full_topo if node in nodes_subset]

    def plot_dag(self, save_dir=None, show: bool = False):
        """Plot the DAG."""
        plt.figure(figsize=(4, 3))
        nx.draw(
            self,
            node_positions=None,
            with_labels=True,
            arrows=True,
            node_size=900,
            node_color="orange",
            font_size=10,
            arrowstyle="->",
            arrowsize=20,
            edge_color="black",
        )
        plt.title("DAG Structure in Topological Order")
        plt.axis("off")
        if save_dir is not None:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            plt.savefig(save_dir)
        if show:
            plt.show()
        plt.close()

    def __repr__(self):
        """Print string representation of the DAG handler."""
        return f"POGPNDAG(nodes={len(self)}, edges={self.number_of_edges()})"


def get_deterministic_order(custom_list: Union[List[str], Set[str]]) -> List[str]:
    """Get the deterministic sort for a list or set of strings."""
    return sorted(custom_list)
