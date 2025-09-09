import torch
from botorch.posteriors import Posterior
from .dag import DAG
from typing import Dict, List, Union
from botorch.models import SingleTaskVariationalGP
from .dag import RegressionNode, RootNode
from .utils import convert_tensor_to_dict, convert_dict_to_tensor


class POGPNPosterior(Posterior):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        node_models_dict: Dict[str, SingleTaskVariationalGP],
        X: torch.Tensor,  # noqa: N803
        dag: DAG,
        dag_nodes: Dict[str, Union[RegressionNode, RootNode]],
        non_root_nodes: List[str],
        root_nodes: List[str],
        deep_nodes: List[str],
        node_parents_dict: Dict[str, List[str]],
        root_node_indices_dict: Dict[str, List[int]],
        objective_node_name: str,
        posterior_transform=None,
    ):
        self.node_models_dict = node_models_dict
        self.dag = dag
        self.dag_nodes = dag_nodes
        self.non_root_nodes = non_root_nodes
        self.root_nodes = root_nodes
        self.deep_nodes = deep_nodes
        self.node_parents_dict = node_parents_dict
        self.objective_node_name = objective_node_name
        self.root_node_indices_dict = root_node_indices_dict

        self.X = X
        self.root_inputs_dict = convert_tensor_to_dict(
            self.X, self.root_node_indices_dict
        )

        self.output_node_indices_dict = self._get_output_node_indices_dict()

        self.posterior_transform = posterior_transform

    def _get_output_node_indices_dict(self):
        list_indices = list(
            range(
                sum(
                    self.dag_nodes[node_name].node_output_dim
                    for node_name in self.non_root_nodes
                )
            )
        )
        output_node_indices_dict = {}
        for node_name in self.non_root_nodes:
            output_node_indices_dict[node_name] = list_indices[
                : self.dag_nodes[node_name].node_output_dim
            ]
            list_indices = list_indices[self.dag_nodes[node_name].node_output_dim :]
        return output_node_indices_dict

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.X.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.X.dtype

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = [
            self.X.shape[-2],
            sum(
                [
                    self.dag_nodes[node_name].node_output_dim
                    for node_name in self.non_root_nodes
                ]
            ),
        ]
        shape = torch.Size(shape)
        return self.batch_shape + shape  # type: ignore

    @property
    def batch_shape(self) -> torch.Size:
        """Compute the batch shape of the GaussianProcessNetwork posterior."""
        return self.X.shape[:-2]

    @property
    def base_sample_shape(self) -> torch.Size:
        """Compute the base sample shape of the GaussianProcessNetwork posterior."""
        return self.event_shape

    def _extended_shape(self, sample_shape: torch.Size) -> torch.Size:
        return sample_shape + self.base_sample_shape  # type: ignore

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return 0, -2

    def _rsample(self, sample_shape=None):
        """Generate samples from the posterior.

        Args:
            sample_shape: The shape of the samples to generate.

        Returns:
            A tensor of shape `sample_shape + event_shape` containing the samples.

        """
        if sample_shape is None:
            sample_shape = torch.Size([])
        base_samples = torch.randn(
            sample_shape + self.base_sample_shape, device=self.device, dtype=self.dtype
        )
        return self._rsample_from_base_samples(sample_shape, base_samples)

    def _rsample_from_base_samples(self, sample_shape=None, base_samples=None):
        """Generate samples from the posterior using provided base samples.

        Args:
            sample_shape: The shape of the samples to generate.
            base_samples: The base samples to use for sampling. If None, new base
                samples will be generated.

        Returns:
            A tensor of shape `sample_shape + event_shape` containing the samples.

        """
        if sample_shape is None:
            sample_shape = torch.Size([])
        if base_samples is None:
            base_samples = torch.randn(
                sample_shape + self.base_sample_shape,
                device=self.device,
                dtype=self.dtype,
            )
        base_samples = base_samples.contiguous()

        base_samples_dict = convert_tensor_to_dict(
            base_samples,
            self.output_node_indices_dict,
        )

        nodes_observation_samples_dict = {}
        nodes_latent_samples_dict = {}

        for node_name in self.dag.get_deterministic_topological_sort_subset(
            self.non_root_nodes
        ):
            if node_name not in self.deep_nodes:
                parent_nodes_data = [
                    self.root_inputs_dict[parent_node_name].clone()
                    for parent_node_name in self.node_parents_dict[node_name]
                ]
                train_X_node_k = torch.cat(parent_nodes_data, dim=-1)  # noqa: N806

                node_model = self.node_models_dict[node_name]
                obs_mvn_at_node = node_model.posterior(
                    train_X_node_k, observation_noise=True
                )
                latent_mvn_at_node = node_model.posterior(
                    train_X_node_k, observation_noise=False
                )

                nodes_observation_samples_dict[node_name] = (
                    obs_mvn_at_node.rsample_from_base_samples(
                        sample_shape,
                        base_samples_dict[node_name].squeeze(-1).contiguous(),
                    )
                )
                nodes_latent_samples_dict[node_name] = (
                    latent_mvn_at_node.rsample_from_base_samples(
                        sample_shape,
                        base_samples_dict[node_name].squeeze(-1).contiguous(),
                    )
                )

            # NOTE: The cloning is kept intact here to not much affect the results of BO. Also only sample_shape expansion is taken care of here for the root nodes and parent node samples. Because the num_task dim expansion is taken care of the in the default posterior method of node model already.

            elif node_name in self.deep_nodes:
                parent_nodes = self.node_parents_dict[node_name]
                parent_samples = []
                for parent_node_name in parent_nodes:
                    if parent_node_name in self.root_nodes:
                        root_data = self.root_inputs_dict[parent_node_name].clone()
                        aux_shape = [sample_shape[0]] + [1] * root_data.ndim
                        parent_samples.append(root_data.repeat(*aux_shape))
                    else:
                        parent_samples.append(
                            nodes_latent_samples_dict[parent_node_name].clone()
                        )

                train_X_node_k = torch.cat(parent_samples, dim=-1)  # noqa: N806

                node_model = self.node_models_dict[node_name]
                obs_mvn_at_node = node_model.posterior(
                    train_X_node_k, observation_noise=True
                )
                latent_mvn_at_node = node_model.posterior(
                    train_X_node_k, observation_noise=False
                )

                nodes_observation_samples_dict[node_name] = (
                    obs_mvn_at_node.rsample_from_base_samples(
                        sample_shape,
                        base_samples_dict[node_name].squeeze(-1).contiguous(),
                    )
                )
                nodes_latent_samples_dict[node_name] = (
                    latent_mvn_at_node.rsample_from_base_samples(
                        sample_shape,
                        base_samples_dict[node_name].squeeze(-1).contiguous(),
                    )
                )

        return nodes_observation_samples_dict

    def rsample_from_base_samples_dict(self, sample_shape=None, base_samples=None):
        """Generate samples dict from the posterior using provided base samples."""
        return self._rsample_from_base_samples(sample_shape, base_samples)

    def rsample_dict(self, sample_shape=None):
        """Generate samples dict from the posterior."""
        return self._rsample(sample_shape=sample_shape)

    def rsample_objective_node(self, sample_shape=None):
        """Generate samples from the posterior for the objective node."""
        return self.rsample_dict(sample_shape=sample_shape)[self.objective_node_name]

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: torch.Tensor
    ) -> torch.Tensor:
        """Generate combined samples tensor from the posterior using provided base samples."""
        nodes_observation_samples_dict = self._rsample_from_base_samples(
            sample_shape, base_samples
        )
        return convert_dict_to_tensor(
            nodes_observation_samples_dict,
            self.output_node_indices_dict,
        )

    def rsample(self, sample_shape=None):
        """Generate combined samples tensor from the posterior."""
        nodes_observation_samples_dict = self._rsample(sample_shape=sample_shape)
        return convert_dict_to_tensor(
            nodes_observation_samples_dict,
            self.output_node_indices_dict,
        )
