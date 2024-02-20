"""
TransferFns: accept and modify a 2d array
"""

import numpy
import copy
import param


class TransferFn(param.Parameterized):
    """
    Function object to modify a matrix in place, e.g. for normalization.

    Used for transforming an array of intermediate results into a
    final version, by cropping it, normalizing it, squaring it, etc.

    Objects in this class must support being called as a function with
    one matrix argument, and are expected to change that matrix in place.
    """
    __abstract = True

    init_keys = param.List(default=[], constant=True, doc="""
        List of item key labels for metadata that that must be
        supplied to the initialize method before the TransferFn may be
        used.""")

    # CEBALERT: can we have this here - is there a more appropriate
    # term for it, general to output functions?  JAB: Please do rename it!
    norm_value = param.Parameter(default=None)

    def initialize(self,  **kwargs):
        """
        Transfer functions may need additional information before the
        supplied numpy array can be modified in place. For instance,
        transfer functions may have state which needs to be allocated
        in memory with a certain size. In other cases, the transfer
        function may need to know about the coordinate system
        associated with the input data.
        """
        if not set(kwargs.keys()).issuperset(self.init_keys):
            raise Exception("TransferFn needs to be initialized with %s"
                            % ','.join(repr(el) for el in self.init_keys))

    def __call__(self,x):
        raise NotImplementedError


class TransferFnWithState(TransferFn):
    """
    Abstract base class for TransferFns that need to maintain a
    self.plastic parameter.

    These TransferFns typically maintain some form of internal history
    or other state from previous calls, which can be disabled by
    override_plasticity_state().
    """

    plastic = param.Boolean(default=True, doc="""
        Whether or not to update the internal state on each call.
        Allows plasticity to be temporarily turned off (e.g for
        analysis purposes).""")

    __abstract = True

    def __init__(self,**params):
        super(TransferFnWithState,self).__init__(**params)
        self._plasticity_setting_stack = []


    def override_plasticity_state(self, new_plasticity_state):
        """
        Temporarily disable plasticity of internal state.

        This function should be implemented by all subclasses so that
        after a call, the output should always be the same for any
        given input pattern, and no call should have any effect that
        persists after restore_plasticity_state() is called.

        By default, simply saves a copy of the 'plastic' parameter to
        an internal stack (so that it can be restored by
        restore_plasticity_state()), and then sets the plastic
        parameter to the given value (True or False).
        """
        self._plasticity_setting_stack.append(self.plastic)
        self.plastic=new_plasticity_state


    def restore_plasticity_state(self):
        """
        Re-enable plasticity of internal state after an
        override_plasticity_state call.

        This function should be implemented by all subclasses to
        remove the effect of the most recent override_plasticity_state call,
        i.e. to reenable changes to the internal state, without any
        lasting effect from the time during which plasticity was disabled.

        By default, simply restores the last saved value of the
        'plastic' parameter.
        """
        self.plastic = self._plasticity_setting_stack.pop()

    def state_push(self):
        """
        Save the current state onto a stack, to be restored with
        state_pop.

        Subclasses must implement state_push and state_pop to store
        state across invocations. The behaviour should be such that
        after state_pop, the state is restored to what it was at
        the time when state_push was called.
        """
        pass

    def state_pop(self):
        """
        Restore the state saved by the most recent state_push call.
        """
        pass



class IdentityTF(TransferFn):
    """
    Identity function, returning its argument as-is.

    For speed, calling this function object is sometimes optimized
    away entirely.  To make this feasible, it is not allowable to
    derive other classes from this object, modify it to have different
    behavior, add side effects, or anything of that nature.
    """

    def __call__(self,x,sum=None):
        pass


class Scale(TransferFn):
    """
    Multiply the input array by some constant factor.
    """

    scale = param.Number(default=1.0, doc="""
         The multiplicative factor that scales the input values.""")

    def __call__(self, x):
        x *= self.scale



class Threshold(TransferFn):
    """
    Forces all values below a threshold to zero, and leaves others unchanged.
    """

    threshold = param.Number(default=0.25, doc="""
        Decision point for determining values to clip.""")

    def __call__(self,x):
        numpy.minimum(x,self.threshold,x)



class BinaryThreshold(TransferFn):
    """
    Forces all values below a threshold to zero, and above it to 1.0.
    """

    threshold = param.Number(default=0.25, doc="""
        Decision point for determining binary value.""")

    def __call__(self,x):
        above_threshold = x>=self.threshold
        x *= 0.0
        x += above_threshold



class DivisiveNormalizeL1(TransferFn):
    """
    TransferFn that divides an array by its L1 norm.

    This operation ensures that the sum of the absolute values of the
    array is equal to the specified norm_value, rescaling each value
    to make this true.  The array is unchanged if the sum of absolute
    values is zero.  For arrays of non-negative values where at least
    one is non-zero, this operation is equivalent to a divisive sum
    normalization.
    """
    norm_value = param.Number(default=1.0)

    def __call__(self,x):
        """L1-normalize the input array, if it has a nonzero sum."""
        current_sum = 1.0*numpy.sum(abs(x.ravel()))
        if current_sum != 0:
            factor = (self.norm_value/current_sum)
            x *= factor



class DivisiveNormalizeL2(TransferFn):
    """
    TransferFn to divide an array by its Euclidean length (aka its L2 norm).

    For a given array interpreted as a flattened vector, keeps the
    Euclidean length of the vector at a specified norm_value.
    """
    norm_value = param.Number(default=1.0)

    def __call__(self,x):
        xr = x.ravel()
        tot = 1.0*numpy.sqrt(numpy.dot(xr,xr))
        if tot != 0:
            factor = (self.norm_value/tot)
            x *= factor



class DivisiveNormalizeLinf(TransferFn):
    """
    TransferFn to divide an array by its L-infinity norm
    (i.e. the maximum absolute value of its elements).

    For a given array interpreted as a flattened vector, scales the
    elements divisively so that the maximum absolute value is the
    specified norm_value.

    The L-infinity norm is also known as the divisive infinity norm
    and Chebyshev norm.
    """
    norm_value = param.Number(default=1.0)

    def __call__(self,x):
        tot = 1.0*(numpy.abs(x)).max()
        if tot != 0:
            factor = (self.norm_value/tot)
            x *= factor



def norm(v,p=2):
    """
    Returns the Lp norm of v, where p is an arbitrary number defaulting to 2.
    """
    return (numpy.abs(v)**p).sum()**(1.0/p)



class DivisiveNormalizeLp(TransferFn):
    """
    TransferFn to divide an array by its Lp-Norm, where p is specified.

    For a parameter p and a given array interpreted as a flattened
    vector, keeps the Lp-norm of the vector at a specified norm_value.
    Faster versions are provided separately for the typical L1-norm
    and L2-norm cases.  Defaults to be the same as an L2-norm, i.e.,
    DivisiveNormalizeL2.
    """
    p = param.Number(default=2)
    norm_value = param.Number(default=1.0)

    def __call__(self,x):
        tot = 1.0*norm(x.ravel(),self.p)
        if tot != 0:
            factor = (self.norm_value/tot)
            x *=factor


class Hysteresis(TransferFnWithState):
    """
    Smoothly interpolates a matrix between simulation time steps, with
    exponential falloff.
    """

    time_constant  = param.Number(default=0.3,doc="""
        Controls the time scale of the interpolation.""")

    def __init__(self,**params):
        super(Hysteresis,self).__init__(**params)
        self.first_call = True
        self.__current_state_stack=[]
        self.old_a = 0

    def __call__(self,x):
        if self.first_call is True:
           self.old_a = x.copy() * 0.0
           self.first_call = False

        new_a = x.copy()
        self.old_a = self.old_a + (new_a - self.old_a)*self.time_constant
        x*=0
        x += self.old_a

    def reset(self):
        self.old_a *= 0

    def state_push(self):
        self.__current_state_stack.append((copy.copy(self.old_a),
                                           copy.copy(self.first_call)))
        super(Hysteresis,self).state_push()

    def state_pop(self):
        self.old_a,self.first_call =  self.__current_state_stack.pop()
        super(Hysteresis,self).state_pop()
