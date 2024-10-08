from skillvla.robomimic.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from skillvla.robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from skillvla.robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from skillvla.robomimic.algo.cql import CQL
from skillvla.robomimic.algo.iql import IQL
from skillvla.robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from skillvla.robomimic.algo.hbc import HBC
from skillvla.robomimic.algo.iris import IRIS
from skillvla.robomimic.algo.td3_bc import TD3_BC
from skillvla.robomimic.algo.diffusion_policy import DiffusionPolicyUNet
