.. _common_issues:

Common Issues
=======================

This section introduces some common issues in PyCFRL and discusses how 
users might deal with these issues.

Non-convergence
-----------------------

Non-convergence happens when the model fails to converge 
at the end of model training. In PyCFRL, non-convergence can arise when

1. Training the :code:`SequentialPreprocessor`.
2. Training the :code:`SimulatedEnvironment`.
3. Training the :code:`FQI`.
4. Training the :code:`FQE` in :code:`evaluate_reward_through_fqe()`.

In theory, non-convergence might lead the models in PyCFRL to produce 
inaccurate outputs. In simulation studies, the models in PyCFRL often perform 
well despite potential non-convergence, though occasionally non-convergence 
does result in notably suboptimal outputs.

Sources of Non-convergence
~~~~~~~~~~~~~~~~~~~~~~~

:code:`SequentialPreprocessor` and :code:`SimulatedEnvironment` rely on a 
neural network or to approximate the underlying environment's transition 
dynamics. Similarly, :code:`FQI` and :code:`FQE` rely 
on a neural network to approximate the Q function in each iteration. 
If the neural network fail to converge during training, then 
non-convergence arises. 

There are many possible reasons why a neural network might fail 
to converge. For example, an inappropriate network depth or learning rate 
might lead to failures to converge. Besides, if the training data is 
noisy or its size is not large enough, the network can also fail to 
converge. Moreover, if the maximum number of training epochs is too small, 
then the training might stop before the network converges, which leads 
the network to fail to converge.

In :code:`FQI` and :code:`FQE`, there is one more source of 
non-convergence. The Q function approximator is updated in each iteration 
of FQI and FQE training. Thus, non-convergence can also arise if the 
approximated Q function fails to converge to the true Q function in these 
iterative updates.

Similarly, many reasons might explain why the approximated Q function 
might fail to converge. One reason might be the maximum number of iteration 
is so small that the training stops before the approximated Q function can 
converge. Also, we note that at each iteration, the approximated Q function 
is obtained by solving a supervised learning problem whose learning target 
depends on (1) the observed rewards and (2) the Q value predicted by the 
approximated Q function from the previous iteration. Therefore, if the 
observed rewards are noisy, or if the Q function is badly approximated 
in previous iterations, then the learning target might be noisy or 
inaccurate, and consequently the approximated Q function might fail to 
converge.

Checking for Neural Network Non-convergence Using Loss Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

PyCFRL provides a simple loss monitoring tool that can check for potential non-convergence 
in the training of neural networks in :code:`SequentialPreprocessor`, 
:code:`SimulatedEnvironment`, :code:`FQI`, and :code:`FQE`. This tool can 
be enabled by setting :code:`is_loss_monitored=True` in 
:code:`evaluate_reward_through_fqe()` and in the constructors of 
:code:`SequentialPreprocessor`, :code:`SimulatedEnvironment`, :code:`FQI`, 
and :code:`FQE`. When this tool is enabled, it divides the training data into 
a training set and a validation set and then monitors the validation loss in 
each epoch of neural network training. At each epoch of neural network 
training, it monitors the percent absolute change in the validation loss. It raises 
a warning if the percent absolute change in the validation loss is greater than some 
threshold in at least one of the final :math:`p` epochs before training stops 
(or is early stopped), where the threshold and :math:`p` are user-specified 
parameters. The threshold is defaulted to :math:`0.005` and :math:`p` is 
defaulted to :math:`10`.

We note that while loss monitoring might flag potential non-convergence, it is 
neither necessary nor sufficient for non-convergence. For example, when the 
validation loss plateaus without reaching the true minimum, loss monitoring will 
not raise warnings despite non-convergence. On the other hand, if the choice of 
the threshold is too small, or the choice of :math:`p` is too large, then the 
tool will raise a warning even though the model has almost converged. Moreover, 
this loss monitoring tool is applicable only when the model type is :code:`"nn"`, 
and it only checks for potential non-convergence in neural network training 
rather than potential non-convergence in the update of the Q function approximators.

Checking for Approximated Q Function Non-convergence Using Q Value Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

PyCFRL also provides a simple Q value monitoring tool that can check for potential non-convergence 
in the training of the approximated Q function in :code:`FQI` and :code:`FQE`. 
This tool can be enabled by setting :code:`is_q_monitored=True` in 
:code:`evaluate_reward_through_fqe()` and in the constructors of :code:`FQI` 
and :code:`FQE`. When this tool is enabled, at each iteration :math:`t > 0`, 
using the current iteration's approximated Q function, it computes 
a vector :math:`Q_t` that contains the approximated Q values at all of the state-action 
pairs that are present in the training trajectory. For each component 
:math:`Q_t^{(i)}` of :math:`Q_t`, it then calculates the percent change

.. math::
    d_i = \frac{|Q_t^{(i)} - Q_{t-1}^{(i)}|}{10^{-8} + |Q_{t-1}^{(i)}|},

where the constant :math:`10^{-8}` is added in the denominator to avoid division by zero. It raises 
a warning if :math:`\max_{i}d_i` is greater than some 
threshold in at least one of the final :math:`r` iterations before the iterative 
model updates stop (or are early stopped), where 
the threshold and :math:`r` are user-specified parameters. The threshold is 
defaulted to :math:`0.005` and :math:`r` is defaulted to :math:`5`.

For reasons similar to those introduced in the previous section, we again 
note that while Q value monitoring might flag potential non-convergence, it 
is neither necessary nor sufficient for non-convergence. Also, it only checks 
for potential non-convergence in the update of the Q function approximators 
rather than potential non-convergence in the neural network training.

Mitigating Non-convergence
~~~~~~~~~~~~~~~~~~~~~~~~

To reduce the likelihood of non-convergence in neural network training, users might 
try increasing the maximum number of training epochs, adjusting the learning rate, 
or increasing the size of the training data. To reduce the likelihood of 
non-convergence in the update of the Q function approximators, users might try 
increasing the maximum number of FQI/FQE iterations or ensuring the training data 
covers a sufficiently large portion of the state and action spaces.