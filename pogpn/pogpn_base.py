import gpytorch
import torch
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskVariationalGP
from botorch.models.model import Model

from torch import Tensor
from botorch.sampling.normal import SobolQMCNormalSampler
from .dag import DAG, RegressionNode, RootNode
import logging
from .utils import convert_tensor_to_dict
from gpytorch.models import GP
from botorch.acquisition.objective import PosteriorTransform
from .pogpn_posterior import POGPNPosterior
from .utils import consolidate_mvn_mixture, consolidate_mtmvn_mixture
from botorch.posteriors import GPyTorchPosterior

from gpytorch.mlls import MarginalLogLikelihood
from linear_operator.operators import LinearOperator
from torch.distributions import Distribution
import pprint
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from botorch.models.transforms.outcome import OutcomeTransform

INDUCING_POINTS_FACTOR = 1.0

logger = logging.getLogger("POGPN BASE")


class _POGPNModelDict(GP):
    def __init__(
        self,
        node_models_dict: Dict[str, SingleTaskVariationalGP],
        dag: DAG,
        dag_nodes: Dict[str, Union[RegressionNode, RootNode]],
        non_root_nodes: List[str],
        root_nodes: List[str],
        deep_nodes: List[str],
        node_parents_dict: Dict[str, List[str]],
        root_node_indices_dict: Dict[str, List[int]],
        num_observations: int,
        device: torch.device,
        dtype: torch.dtype,
        masks_dict: Optional[Dict[str, Tensor]] = None,
    ):
        super().__init__()
        # Register the node models as submodules so they appear in the summary
        self.node_models_dict = node_models_dict
        for name, module in node_models_dict.items():
            self.add_module(f"node_model_{name}", module)

        self.dag = dag
        self.dag_nodes = dag_nodes
        self.non_root_nodes = non_root_nodes
        self.root_nodes = root_nodes
        self.deep_nodes = deep_nodes
        self.node_parents_dict = node_parents_dict
        self.root_node_indices_dict = root_node_indices_dict
        self.num_observations = num_observations
        self.device = device
        self.dtype = dtype
        self.masks_dict = masks_dict if masks_dict is not None else {}
        self.train_inputs: List[Any] = []
        self.train_targets: Any = None

        self.node_mlls_dict = {}
        # self.node_mlls_dict = torch.nn.ModuleDict({})
        for node_name in self.non_root_nodes:
            if isinstance(self.dag_nodes[node_name], RegressionNode):
                self.node_mlls_dict[node_name] = self.dag_nodes[node_name].node_mll  # type: ignore
            else:
                raise ValueError(
                    f"Node {node_name} is not a RegressionNode. Please provide a MarginalLogLikelihood for all non-root nodes."
                )

    def forward(self, X: Tensor):  # noqa: N803
        """Forward pass of the POGPNPathwise model."""
        root_input_dict = convert_tensor_to_dict(
            X,
            self.root_node_indices_dict,
        )
        samples_for_child_nodes_dict = {}
        latent_mvn_at_node_dict: Dict[
            str, Union[Tensor, Distribution, LinearOperator]
        ] = {}

        for node_name in self.non_root_nodes:
            samples_for_child_nodes_dict[node_name] = torch.empty(
                torch.Size(
                    [
                        gpytorch.settings.num_likelihood_samples.value(),  # type: ignore
                        self.num_observations,
                        self.dag_nodes[node_name].node_output_dim,
                    ]
                )  # type: ignore
            ).to(self.device, self.dtype)  # type: ignore

        for node_name in self.dag.get_deterministic_topological_sort_subset(
            self.non_root_nodes
        ):
            regression_node = self.dag_nodes[node_name]
            assert isinstance(regression_node, RegressionNode), (
                f"Node {node_name} must be a RegressionNode"
            )
            node_model = regression_node.node_model
            node_sampler = regression_node.node_sampler

            if (
                node_sampler is not None
                and node_name not in self.deep_nodes
                and not node_sampler.sample_shape[0]
                == gpytorch.settings.num_likelihood_samples.value()
            ):
                raise ValueError(
                    f"Node {node_name} has a sampler with a sample shape of {node_sampler.sample_shape[0]} but the number of likelihood samples is {gpytorch.settings.num_likelihood_samples.value()}. Please put the POGPN model initialization, calling and posterior within the gpytorch.settings.num_likelihood_samples context."
                )

            if node_model is None:
                raise ValueError(
                    f"Node {node_name} has no model. Please provide a model for all non-root nodes."
                )
            elif node_sampler is None:
                raise ValueError(
                    f"Node {node_name} has no sampler. Please provide a sampler for all non-root nodes."
                )

            if node_name not in self.deep_nodes:
                parent_observations = []

                for parent_node_name in self.node_parents_dict[node_name]:
                    parent_observations.append(root_input_dict[parent_node_name])

                train_X_node: Tensor = torch.cat(parent_observations, dim=-1)  # noqa: N806
                latent_mvn_at_node_dict[node_name] = node_model(train_X_node)

                samples_for_child_nodes_dict[node_name] = node_sampler(
                    GPyTorchPosterior(latent_mvn_at_node_dict[node_name])  # type: ignore
                )

            elif node_name in self.deep_nodes:
                parent_nodes = self.node_parents_dict[node_name]
                parent_observations = []
                node_output_dim = self.dag_nodes[node_name].node_output_dim
                mc_samples = gpytorch.settings.num_likelihood_samples.value()
                for parent_node_name in parent_nodes:
                    if parent_node_name in self.root_nodes:
                        obs = root_input_dict[parent_node_name].clone()
                        if node_output_dim > 1:
                            aux_shape = [mc_samples, node_output_dim] + [1] * obs.ndim
                            obs = obs.unsqueeze(-3).unsqueeze(-4).repeat(*aux_shape)
                        else:
                            aux_shape = [mc_samples] + [1] * obs.ndim
                            obs = obs.unsqueeze(-3).repeat(*aux_shape)  # type: ignore
                        parent_observations.append(obs)
                    else:
                        parent_samples = samples_for_child_nodes_dict[parent_node_name]
                        if node_output_dim > 1:
                            aux_shape = (
                                [1] * (parent_samples.ndim - 2)
                                + [node_output_dim]
                                + [1] * 2
                            )
                            parent_samples = parent_samples.unsqueeze(-3).repeat(
                                *aux_shape
                            )
                        parent_observations.append(parent_samples)

                train_X_node = torch.cat(parent_observations, dim=-1)  # noqa: N806

                batached_latent_mvn = node_model(train_X_node)

                # latent_mvn_at_node_dict[node_name] = batached_latent_mvn

                if isinstance(
                    batached_latent_mvn,
                    gpytorch.distributions.MultitaskMultivariateNormal,
                ):
                    latent_mvn_at_node_dict[node_name] = consolidate_mtmvn_mixture(
                        batached_latent_mvn
                    )
                else:
                    latent_mvn_at_node_dict[node_name] = consolidate_mvn_mixture(
                        batached_latent_mvn
                    )

                samples_for_child_nodes_dict[node_name] = node_sampler(
                    GPyTorchPosterior(latent_mvn_at_node_dict[node_name])  # type: ignore
                )

        return latent_mvn_at_node_dict

    def __call__(
        self, *inputs, **kwargs
    ) -> Dict[str, Union[Tensor, Distribution, LinearOperator]]:
        outputs = self.forward(*inputs, **kwargs)
        if isinstance(outputs, dict):
            return {
                key: _validate_module_outputs(output) for key, output in outputs.items()
            }
        else:
            raise RuntimeError(f"Expected dict output, got {type(outputs)}")

    def train_pogpn_nodewise(
        self, optimizer: Callable, optimizer_kwargs: Dict[str, Any]
    ):
        """Train the POGPNNodewise model."""
        samples_for_child_nodes_dict: Dict[str, Tensor] = {}

        for node_name in self.non_root_nodes:
            samples_for_child_nodes_dict[node_name] = torch.empty(
                torch.Size(
                    [
                        gpytorch.settings.num_likelihood_samples.value(),  # type: ignore
                        self.num_observations,
                        self.dag_nodes[node_name].node_output_dim,
                    ]
                )  # type: ignore
            ).to(self.device, self.dtype)  # type: ignore

        for node_name in self.dag.get_deterministic_topological_sort_subset(
            self.non_root_nodes
        ):
            regression_node = self.dag_nodes[node_name]
            assert isinstance(regression_node, RegressionNode), (
                f"Node {node_name} must be a RegressionNode"
            )
            node_model = self.node_models_dict[node_name]
            node_sampler = regression_node.node_sampler

            if (
                node_sampler is not None
                # and node_name not in self.deep_nodes
                and not node_sampler.sample_shape[0]
                == gpytorch.settings.num_likelihood_samples.value()
            ):
                raise ValueError(
                    f"Node {node_name} has a sampler with a sample shape of {node_sampler.sample_shape[0]} but the number of likelihood samples is {gpytorch.settings.num_likelihood_samples.value()}. Please put the POGPN model initialization, calling and posterior within the gpytorch.settings.num_likelihood_samples context."
                )

            if node_model is None:
                raise ValueError(
                    f"Node {node_name} has no model. Please provide a model for all non-root nodes."
                )
            elif node_sampler is None:
                raise ValueError(
                    f"Node {node_name} has no sampler. Please provide a sampler for all non-root nodes."
                )

            if node_name not in self.deep_nodes:
                fit_gpytorch_mll(
                    self.node_mlls_dict[node_name],
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                )  # type: ignore

                latent_mvn_at_node = node_model.posterior(
                    X=node_model.model.train_inputs[0], observation_noise=False
                )

                samples_for_child_nodes_dict[node_name] = node_sampler(
                    latent_mvn_at_node
                ).detach()
            elif node_name in self.deep_nodes:
                parent_nodes = self.node_parents_dict[node_name]
                parent_observations = []
                for parent_node_name in parent_nodes:
                    if parent_node_name in self.root_nodes:
                        node_obs = self.dag_nodes[
                            parent_node_name
                        ].node_observation.clone()  # type: ignore
                        assert node_obs is not None, (
                            f"Node observation for {parent_node_name} is None"
                        )
                        aux_shape = [
                            gpytorch.settings.num_likelihood_samples.value()
                        ] + [1] * node_obs.ndim  # type: ignore
                        parent_observations.append(
                            node_obs.unsqueeze(-3).repeat(*aux_shape)  # type: ignore
                        )
                    else:
                        parent_observations.append(
                            samples_for_child_nodes_dict[parent_node_name].clone()
                        )

                train_X_node_k = torch.cat(parent_observations, dim=-1)  # noqa: N806

                # SingleTaskVariationalGP expects train_inputs to be a list
                node_model.model.train_inputs = [train_X_node_k]  # type: ignore

                fit_gpytorch_mll(
                    self.node_mlls_dict[node_name],
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                )  # type: ignore

                latent_mvn_at_node = node_model.posterior(
                    X=train_X_node_k, observation_noise=False
                )

                if isinstance(
                    latent_mvn_at_node.distribution,
                    gpytorch.distributions.MultitaskMultivariateNormal,
                ):
                    latent_mvn_at_node = consolidate_mtmvn_mixture(latent_mvn_at_node)
                else:
                    latent_mvn_at_node = consolidate_mvn_mixture(latent_mvn_at_node)

                samples_for_child_nodes_dict[node_name] = node_sampler(
                    latent_mvn_at_node
                ).detach()


def _validate_module_outputs(outputs):
    if isinstance(outputs, tuple):
        if not all(
            isinstance(output, (Tensor, Distribution, LinearOperator))
            for output in outputs
        ):
            raise RuntimeError(
                "All outputs must be a torch.Tensor, Distribution, or LinearOperator. "
                "Got {}".format([output.__class__.__name__ for output in outputs])
            )
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
    elif isinstance(outputs, (Tensor, Distribution, LinearOperator)):
        return outputs
    else:
        raise RuntimeError(
            "Output must be a torch.Tensor, Distribution, or LinearOperator. Got {}".format(
                outputs.__class__.__name__
            )
        )


class POGPNBase(Model):
    """POGPNBase with node-wise conditional training of the nodes."""

    def __init__(
        self,
        dag: DAG,
        data_dict: Dict[str, Tensor],
        root_node_indices_dict: Dict[str, List[int]],
        objective_node_name: str,
        inducing_point_ratio: float = 1.0,
        use_rbf_kernel: bool = True,
        mll_beta: float = 1.0,
        mll_type: str = "ELBO",
        seed: Optional[int] = None,
        masks_dict: Optional[Dict[str, Tensor]] = None,
    ):
        """Initialize the POGPNBase model.

        Args:
            dag: The DAG of the model.
            data_dict: Dictionary of node names and their corresponding observations.
                data_dict should have the combined tensor of all input nodes as the key 'inputs'
                and the respective observed outputs as the keys.
                Ex: for x1->y1, x1,x2-> y2, y1,y2->y3, (x1, x2 are each a tensor of shape (N, 2) and y1, y2, y3 are each a tensor of shape (N, D_y*)). Where D_y* is the -1 dimension of each output node.
                data_dict should have the keys 'inputs', 'y1', 'y2', 'y3'.
                'inputs' should be a tensor concatenated tensor of all the root nodes shaped (N, 4).
                The order of the nodes in the 'inputs' tensor should be the same as the (node_name, indices) pairs specified in the 'root_node_indices_dict'.
                'y1' should be a tensor
                'y2' should be a tensor
                'y3' should be a tensor
                Each output tensor can be of different dimensions (one or multiple outputs)
            root_node_indices_dict: Dictionary of root node names and their corresponding indices.
                Ex: for x1->y1, x1,x2-> y2, y1,y2->y3,
                root_node_indices_dict should be {'x1': [0, 1], 'x2': [2, 3]}
                where 0 and 1 are the indices of x1 and x2 respectively if "inputs" is a tensor of shape (N, 4).
            inducing_point_ratio: The ratio of inducing points to the number of observations.
                This is used to determine the number of inducing points for each node.
            use_rbf_kernel: Whether to use an `RBFKernel`. If False, uses a `MaternKernel`.
            mll_beta: The beta parameter for the ApproximateMarginalLogLikelihood.
            mll_type: The type of marginal log likelihood to use.
                Can be "ELBO" for Variational ELBO or "PLL" for Predictive Log Likelihood.
            previous_node_GPs: Dictionary of node names and their corresponding GP models.
                This has been provided to be able to use GreedyImprovementReduction for the Inducing Points if needed.
            objective_node_name: The name of the node that is the objective.
            seed: The seed for the random number generator.

        """
        super().__init__()
        self.dag = dag
        self.objective_node_name = objective_node_name
        self.dag_nodes: Dict[str, Union[RegressionNode, RootNode]] = {
            node: self.dag.nodes[node]["data"] for node in self.dag.nodes()
        }

        self.seed = (
            seed if seed is not None else int(torch.randint(0, 1000000, (1,)).item())
        )
        self.root_node_indices_dict = root_node_indices_dict
        self.inducing_point_ratio = inducing_point_ratio
        self.use_rbf_kernel = use_rbf_kernel
        self.mll_beta = mll_beta
        self.mll_type = mll_type
        self.masks_dict = masks_dict

        device = []
        dtype = []

        num_observations = []

        root_input_tensor_dict = convert_tensor_to_dict(
            data_dict["inputs"], self.root_node_indices_dict
        )
        data_dict.update(root_input_tensor_dict)

        for node in self.dag_nodes:
            data = data_dict.get(node, None)
            if data is None:
                raise ValueError(
                    f"Node {node} has no observations. Please provide observations for all nodes."
                )
            else:
                self.dag_nodes[node].node_observation = data

                num_observations.append(data.shape[-2])
                device.append(data.device)
                dtype.append(data.dtype)

        if len(set(device)) > 1 or len(set(dtype)) > 1:
            raise ValueError(
                "All node observations must have the same device and dtype."
            )
        if len(set(num_observations)) > 1:
            raise ValueError(
                "All node observations must have the same number of observations."
            )
        self.num_observations = num_observations[0]
        logger.debug(f"Number of observations: {self.num_observations}")

        self.device = device[0]
        self.dtype = dtype[0]

        self.root_nodes: List[str] = self.dag.root_nodes
        logger.debug(f"Root nodes: {self.root_nodes}")

        self.non_root_nodes: List[str] = self.dag.nodes_without_root_nodes(
            self.dag_nodes
        )
        logger.debug(f"Non-root nodes: {self.non_root_nodes}")

        self.deep_nodes: List[str] = self.dag.get_deep_nodes(self.non_root_nodes)
        logger.debug(f"Deep nodes: {self.deep_nodes}")

        # Pre-compute parents dictionary for faster lookups
        self.node_parents_dict = {
            node_name: self.dag.get_node_parents(node_name)
            for node_name in self.non_root_nodes
        }
        logger.debug(
            f"Node parents dictionary:\n{pprint.pformat(self.node_parents_dict, indent=2)}"
        )

        self._initialize_node_models()

        logger.debug(
            f"Root node indices dictionary:\n{pprint.pformat(self.root_node_indices_dict, indent=2)}"
        )

        node_models_dict = {}
        for node_name in self.non_root_nodes:
            node = self.dag_nodes[node_name]
            if isinstance(node, RegressionNode) and node.node_model is not None:
                node_models_dict[node_name] = node.node_model

        self.model = _POGPNModelDict(
            node_models_dict=node_models_dict,
            dag=self.dag,
            dag_nodes=self.dag_nodes,
            non_root_nodes=self.non_root_nodes,
            root_nodes=self.root_nodes,
            deep_nodes=self.deep_nodes,
            node_parents_dict=self.node_parents_dict,
            root_node_indices_dict=self.root_node_indices_dict,
            num_observations=self.num_observations,
            device=self.device,
            dtype=self.dtype,
            masks_dict=self.masks_dict,
        )

        # Store likelihood references without creating a ModuleDict to avoid recursive parameter counting
        self.likelihood = {
            node_name: self.dag_nodes[node_name].node_model.likelihood  # type: ignore
            for node_name in self.non_root_nodes
        }

    def train(self):
        """Set the model to training mode."""
        for node_name in self.non_root_nodes:
            self.dag_nodes[node_name].node_model.train()  # type: ignore

    def eval(self):
        """Set the model to evaluation mode."""
        for node_name in self.non_root_nodes:
            self.dag_nodes[node_name].node_model.eval()  # type: ignore

    def to(self, device: torch.device, dtype: torch.dtype):
        """Move the model to the given device and dtype."""
        super().to(device, dtype)
        for node_name in self.non_root_nodes:
            self.dag_nodes[node_name].node_model.to(device, dtype)  # type: ignore

    def _initialize_node_models(
        self,
    ):
        """Initialize the node models.

        Args:
            inducing_point_allocators: Dictionary of node names and their corresponding inducing point allocators.
                This has been provided to be able to use GreedyImprovementReduction for the Inducing Points if needed.
            learn_inducing_points: Whether to learn the inducing points.

        """
        logger.debug("Initializing node models...")
        for node_name in self.dag.get_deterministic_topological_sort_subset(
            self.non_root_nodes
        ):
            parent_observations = []

            for parent in self.node_parents_dict[node_name]:
                parent_observations.append(self.dag_nodes[parent].node_observation)

            train_X_node = torch.cat(parent_observations, dim=-1).clone()  # noqa: N806

            train_Y_node = self.dag_nodes[node_name].node_observation.clone()  # type: ignore # noqa: N806

            if isinstance(self.dag_nodes[node_name], RegressionNode):
                node_observation_noise = self.dag_nodes[
                    node_name
                ].node_observation_noise  # type: ignore
            else:
                node_observation_noise = None

            node_transform = self.dag_nodes[node_name].node_transform  # type: ignore

            inducing_point_allocator = self.dag_nodes[
                node_name
            ].inducing_point_allocator  # type: ignore
            learn_inducing_points = self.dag_nodes[node_name].learn_inducing_points  # type: ignore

            model, mll = self._get_node_model_and_mll(
                node_name=node_name,
                train_X_node=train_X_node,
                train_Y_node=train_Y_node,  # type: ignore
                inducing_point_ratio=self.inducing_point_ratio,
                use_rbf_kernel=self.use_rbf_kernel,
                mll_beta=self.mll_beta,
                mll_type=self.mll_type,
                inducing_point_allocator=inducing_point_allocator,
                learn_inducing_points=learn_inducing_points,
                node_observation_noise=node_observation_noise,
                node_transform=node_transform,
            )
            if self.masks_dict is not None and node_name in self.masks_dict:
                mll.row_mask = self.masks_dict[node_name]
            mll.train()

            self.dag_nodes[node_name].node_model = model  # type: ignore
            self.dag_nodes[node_name].node_mll = mll  # type: ignore

            # if node_name in self.deep_nodes:
            #     self.dag_nodes[node_name].node_sampler = SobolQMCNormalSampler(  # type: ignore
            #         sample_shape=torch.Size([]),
            #         seed=self.seed,
            #     ).to(self.device, self.dtype)
            # else:
            self.dag_nodes[node_name].node_sampler = SobolQMCNormalSampler(  # type: ignore
                sample_shape=torch.Size(
                    [gpytorch.settings.num_likelihood_samples.value()]  # type: ignore
                ),
                seed=self.seed,
            ).to(self.device, self.dtype)

            logger.debug(
                f"{node_name}: Likelihood = {self.dag_nodes[node_name].node_mll.likelihood.__class__.__name__}"  # type: ignore
            )
            logger.debug(
                f"{node_name}: Model = {self.dag_nodes[node_name].node_mll.model.__class__.__name__}"  # type: ignore
            )
            try:
                if hasattr(
                    self.dag_nodes[node_name].node_mll.model.variational_strategy,  # type: ignore
                    "inducing_points",
                ):
                    num_inducing_points = self.dag_nodes[
                        node_name
                    ].node_mll.model.variational_strategy.inducing_points.shape[-2]  # type: ignore
                elif hasattr(
                    self.dag_nodes[node_name].node_mll.model.variational_strategy,  # type: ignore
                    "base_variational_strategy",
                ):
                    num_inducing_points = self.dag_nodes[
                        node_name
                    ].node_mll.model.variational_strategy.base_variational_strategy.inducing_points.shape[  # type: ignore
                        -2
                    ]
                else:
                    num_inducing_points = None
                logger.debug(
                    f"{node_name}: Number of inducing points = {num_inducing_points}"
                )
            except Exception as e:
                logger.debug(
                    f"{node_name}: Error getting number of inducing points: {e}"
                )

    @abstractmethod
    def _get_node_model_and_mll(
        self,
        node_name: str,
        train_X_node: Tensor,  # noqa: N803
        train_Y_node: Tensor,  # noqa: N803
        inducing_point_ratio: float,
        use_rbf_kernel: bool,
        mll_beta: float,
        mll_type: str,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        learn_inducing_points: bool = True,
        node_observation_noise: Optional[float] = None,
        node_transform: Optional[OutcomeTransform] = None,
    ) -> Tuple[Model, MarginalLogLikelihood]:
        """Get the node model and marginal log likelihood for the given node."""
        raise NotImplementedError(
            "Get node model and marginal log likelihood not implemented for POGPNBase."
        )

    @abstractmethod
    def forward(self, X: Tensor):  # noqa: N803
        """Forward pass of the POGPNBase model."""
        raise NotImplementedError("Forward pass not implemented for POGPNBase.")

    def posterior(
        self,
        X: Tensor,  # noqa: N803
        observation_noise: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
    ):
        """Posterior of the POGPNBase model."""
        node_models_dict: Dict[str, SingleTaskVariationalGP] = {}
        for node_name in self.non_root_nodes:
            node = self.dag_nodes[node_name]
            if isinstance(node, RegressionNode) and node.node_model is not None:
                node_models_dict[node_name] = node.node_model

        return POGPNPosterior(
            node_models_dict=node_models_dict,
            X=X,
            dag=self.dag,
            dag_nodes=self.dag_nodes,
            non_root_nodes=self.non_root_nodes,
            root_nodes=self.root_nodes,
            deep_nodes=self.deep_nodes,
            node_parents_dict=self.node_parents_dict,
            root_node_indices_dict=self.root_node_indices_dict,
            objective_node_name=self.objective_node_name,
            posterior_transform=posterior_transform,
        )
