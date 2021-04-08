import torch
import random
from abc import abstractclassmethod,ABC
from typing import Sequence

# 继承ABC是继承抽象基类，含有@abstractclassmethod是抽象方法，只能被重写，类是抽象类，不可被实例化
class BaseStrategy(ABC):
    def __call__(self,weights,amount=0.0):
        '''
        amount:裁剪的通道数比例
        '''
        return self.apply(weights,amount=amount)
    
    @abstractclassmethod
    def apply(self,weights,amount=0.0) -> Sequence[int]: # return index
        raise NotImplementedError

class RandomStrategy(BaseStrategy):
    def apply(self,weights,amount=0.0) -> Sequence[int]:
        assert amount>=0 and amount<1,"error:amount >= 0.0"
        len_weight = len(weights)
        return random.sample(list(range(len_weight)),k=int(amount*len_weight)) # # 保证至少要有一个通道保留

class LNStrategy(BaseStrategy):
    def __init__(self,p):
        self.p = p # 用于选择l1 norm 还是 l2 norm
    
    def apply(self,weights,amount=0.0) -> Sequence[int]:
        assert amount>=0 and amount<1,"error:amount >= 0.0"
        len_weight = len(weights)
        ln_norm = torch.norm(weights.view(len_weight,-1),p=self.p,dim=1)
        n_to_prune = int(amount*len_weight) # 保证至少要有一个通道保留
        if n_to_prune == 0:
            return []
        #  y, i = torch.kthvalue(x, k, n) 沿着n维度返回第k小的数据
        threshold = torch.kthvalue(ln_norm, k=n_to_prune).values
        # 返回非零元素的索引，二维返回两个，三维返回三值....
        indices = torch.nonzero(ln_norm <= threshold).view(-1).tolist()
        return indices

class L1Strategy(LNStrategy):
    '''
    channel prune 常用的通道选择策略
    '''
    def __init__(self):
        super(L1Strategy, self).__init__(p=1)

class L2Strategy(LNStrategy):
    def __init__(self):
        super(L2Strategy, self).__init__(p=2)