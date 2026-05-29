.. _runtime_warnings_and_errors:

Runtime Warnings and Errors
==============================

This section discusses some common runtime warnings and errors that might occur in :code:`PyCFRL`.

Unstable Loss Warning
-------------------------------

When training :code:`SequentialPreprocessor`, :code:`SimulatedEnvironment`, :code:`FQI`, or :code:`FQE`, the following warning might be raised: :code:`The fluctuation in the loss is not small enough in at least one of the final [p] epochs during neural network training`.
This warning is a result of loss monitoring, and it is triggered if the percent absolute change in the validation loss during neural network training is greater than some threshold in at least one of the final :math:`p` epochs before all training epochs are finished, where the threshold and :math:`p` are user-specified parameters.
This warning aims to flag potential non-convergence during neural network training, but it is neither necessary nor suffcient for such non-convergence.
Please see :ref:`Common Issues <common_issues>` for more information on non-convergence and loss monitoring.

**What to do about this warning:** This warning might be raised simply because the chosen threshold and :math:`p` are not reasonable.
The threshold and :math:`p` are specifed by :code:`loss_monitoring_min_delta` and :code:`loss_monitoring_patience`, respectively, in :code:`evaluate_reward_through_fqe()` and in the constructors of :code:`SequentialPreprocessor`, :code:`SimulatedEnvironment`, :code:`FQI`, and :code:`FQE`.
Adjusting to a looser threshold or a smaller :code:`p` might resolve the warning. 
On the other hand, if non-convergence is indeed the culprit, then increasing the maximum number of training epochs, adjusting the learning rate, and/or increasing the size of the training data might help mitigate non-convergence and therefore resolve this warning.

Unstable Q Value Warning
-------------------------------

When training :code:`FQI` or :code:`FQE`, the following warning might be raised: :code:`The fluctuation in the Q values is not small enough in at least one of the final [r] iterations during FQI training` or :code:`The fluctuation in the Q values is not small enough in at least one of the final [r] iterations during FQE training`.
This warning is a result of Q value monitoring, and it is triggered if the change in the Q value predicted by the estimated Q function in FQI or FQE is greater than some threshold in at least one of the final :math:`r` iterations before all training iterations are completed.
This warning aims to flag potential non-convergence during the iterative Q function model updates in FQI and FQE training, though it is neither necessary nor sufficient fo such non-convergence.
Please see :ref:`Common Issues <common_issues>` for more information on non-convergence and Q value monitoring.

**What to do about this warning:** This warning might be raised simply because the chosen threshold and :math:`r` are not reasonable.
The threshold and :math:`r` are specified by :code:`q_monitoring_min_delta` and :code:`q_monitoring_patience` in :code:`evaluate_reward_through_fqe()` and in the constructors of :code:`FQI` and :code:`FQE`.
Adjusting to a looser threshold or a smaller :code:`r` might resolve the warning.
On the other hand, if non-convergence is indeed the culprit, then increasing the maximum number of FQI/FQE iterations and/or ensuring the training data covers a sufficiently large portion of the state and action spaces might help mitigate non-convergence and therefore resolve this warning.