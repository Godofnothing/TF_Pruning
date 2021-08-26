import math

from abc import abstractmethod

class PruningSchedule:

    def __init__(self, begin_epoch, end_epoch, frequency, **kwargs):
        self._validate_params(begin_epoch, end_epoch, frequency)

        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.frequency = frequency
        
        self.current_epoch = 0

    def _validate_params(self, begin_epoch, end_epoch, frequency):
        if begin_epoch < 0:
            raise ValueError("begin step cannot be negative number")

        if frequency <= 0:
            raise ValueError('frequency has to be nonnegative')

        if begin_epoch > end_epoch:
            raise ValueError("begin_epoch cannot be larger than end step")

    @staticmethod
    def _validate_sparsity(sparsity):
        assert 0.0 <= sparsity <= 1.0, "sparsity has to lie in range [0, 1]"

    @abstractmethod
    def get_sparsity(self)->float:
        pass

    def is_prune_epoch(self)->bool:
        return (self.current_epoch - self.begin_epoch) % self.frequency == 0


class ConstantSparsity(PruningSchedule):

    def __init__(
        self, 
        sparsity,
        frequency,
        begin_epoch=0, 
        end_epoch=-1
    ):
        '''
        args:
            sparsity: float 
        '''
        super(ConstantSparsity, self).__init__(begin_epoch, end_epoch, frequency)
        self._validate_sparsity(sparsity)
        self.sparsity = sparsity

    def get_sparsity(self)->float:
        if self.current_epoch >= self.begin_epoch:
            return self.sparsity
        else:
            return 1.0


class PolynomialDecay(PruningSchedule):

    def __init__(
        self, 
        init_sparsity,
        final_sparsity,
        frequency,
        power=2,
        begin_epoch=0, 
        end_epoch=-1
    ):
        '''
        args:
            sparsity: float 
        '''
        super(PolynomialDecay, self).__init__(begin_epoch, end_epoch, frequency)

        assert end_epoch > begin_epoch, f"{self.__class__.__name__} requires end_epoch > begin_epoch"

        self._validate_sparsity(init_sparsity)
        self._validate_sparsity(final_sparsity)
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.power = power

    def get_sparsity(self)->float:
        if self.current_epoch < self.begin_epoch:
            return 1.0

        cur_offset = min(self.current_epoch, self.end_epoch) - self.begin_epoch
        end_offset = self.end_epoch - self.begin_epoch

        return (self.final_sparsity - self.init_sparsity) * (1 - cur_offset / end_offset) ** self.power + self.final_sparsity


class ExponentialDecay(PruningSchedule):

    def __init__(
        self, 
        init_sparsity,
        final_sparsity,
        frequency,
        begin_epoch=0, 
        end_epoch=-1
    ):
        '''
        args:
            init_sparsity: float 
            final_sparsity: float
        '''
        super(ExponentialDecay, self).__init__(begin_epoch, end_epoch, frequency)

        assert end_epoch > begin_epoch, f"{self.__class__.__name__} requires end_epoch > begin_epoch"

        self._validate_sparsity(init_sparsity)
        self._validate_sparsity(final_sparsity)
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        
        self.exp_base = math.log(self.final_sparsity / self.init_sparsity) / (end_epoch - begin_epoch)

    def get_sparsity(self)->float:
        if self.current_epoch < self.begin_epoch:
            return 1.0

        cur_offset = min(self.current_epoch, self.end_epoch) - self.begin_epoch

        return self.init_sparsity * math.exp(cur_offset * self.exp_base)
    

class CosineAnnealingDecay(PruningSchedule):

    def __init__(
        self, 
        init_sparsity,
        final_sparsity,
        frequency,
        begin_epoch=0, 
        end_epoch=-1
    ):
        '''
        args:
            sparsity: float 
        '''
        super(CosineAnnealingDecay, self).__init__(begin_epoch, end_epoch, frequency)

        assert end_epoch > begin_epoch, f"{self.__class__.__name__} requires end_epoch > begin_epoch"

        self._validate_sparsity(init_sparsity)
        self._validate_sparsity(final_sparsity)
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        
        self.cosine_alpha = math.pi / (end_epoch - begin_epoch)

    def get_sparsity(self)->float:
        if self.current_epoch < self.begin_epoch:
            return 1.0

        cur_offset = min(self.current_epoch, self.end_epoch) - self.begin_epoch

        return 0.5 * (self.init_sparsity - self.final_sparsity) * (1 + math.cos(self.cosine_alpha * cur_offset)) + \
            self.final_sparsity
