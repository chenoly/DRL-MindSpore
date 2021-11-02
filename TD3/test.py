from Model import Critic
from mindspore import Tensor
from mindspore import load_param_into_net
import copy
C1 = Critic(state_dim=2, action_dim=1)
C2 = Critic(state_dim=2, action_dim=1)
# C1.load_parameter_slice(C2.parameters_dict())
# load_param_into_net(C1, C2.parameters_dict())
c1_ = C1.parameters_dict()
c2_ = C2.parameters_dict()
for p, p1 in zip(c1_, c2_):
    print(Tensor(c1_[p]))
    print(Tensor(c2_[p1]))
    print(c2_[p1].clone())
