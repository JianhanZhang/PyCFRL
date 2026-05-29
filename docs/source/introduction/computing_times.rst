Computing Times
========================

To provide a reference of the approximated computing time required by PyCFRL, we 
timed three of the four workflows introduced in the 
:ref:`"Example Workflows" <example_workflows>` section 
under various combinations of the number of training samples (:math:`N`) and the 
number of transitions (:math:`T`): the preprocessing only workflow, the policy 
learning only workflow, and the real data workflow. Each workflow was run for 
10 repetitions under each :math:`N` and :math:`T`, and the mean and standard 
deviation (in parenthesis) of the computing times are recorded in the tables 
below. Early stopping and non-convergence checking were not enabled during these 
experiments. The unit of the computing times here is second (s).

The timing experiments were run on an internal cluster equipped with two 36-core 
Intel Xeon Gold 6154 CPUs clocked at 3.0 GHz and 187 GB of RAM. Each job was 
executed via SLURM on a single node using 1 CPU core and 1 GB of memory. The 
version of :code:`PyCFRL` used in the timing experiments is version 0.1.0. Please 
note that the computing times in this section are for reference only. The 
actual computing times often vary by the computing hardware and the amount of 
parallel computing tasks. Also, we only recorded the computing time for full 
workflows. For advanced users interested in the computing times of individual 
functions, we recommend using profiling tools such as :code:`cProfile` and 
:code:`pyinstrument`.

The Preprocessing Only Workflow
------------------------

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 20

   * - :math:`N`, :math:`T`
     - :math:`T=10`
     - :math:`T=20`
   * - :math:`N=100`
     - :math:`32.1 \text{ } (0.15)`
     - :math:`64.3 \text{ } (0.29)`
   * - :math:`N=500`
     - :math:`154.3 \text{ } (0.69)`
     - :math:`309.9 \text{ } (1.75)`
   * - :math:`N=1000`
     - :math:`311.2 \text{ } (1.65)`
     - :math:`615.3 \text{ } (1.91)`

Code used for timing: See `here <https://github.com/JianhanZhang/CFRL/blob/main/examples/workflow_computing_times/time_preprocessing_only_workflow.py>`_.

The Preprocessing + Policy Learning Workflow
------------------------

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 20

   * - :math:`N`, :math:`T`
     - :math:`T=10`
     - :math:`T=20`
   * - :math:`N=100`
     - :math:`58.2 \text{ } (0.58)`
     - :math:`92.8 \text{ } (1.61)`
   * - :math:`N=500`
     - :math:`172.1 \text{ } (3.59)`
     - :math:`333.9 \text{ } (13.00)`
   * - :math:`N=1000`
     - :math:`326.5 \text{ } (4.17)`
     - :math:`637.2 \text{ } (9.51)`

Code used for timing: See `here <https://github.com/JianhanZhang/CFRL/blob/main/examples/workflow_computing_times/time_preprocessing_policy_learning_workflow.py>`_.

The Real Data Workflow
------------------------

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 20

   * - :math:`N`, :math:`T`
     - :math:`T=10`
     - :math:`T=20`
   * - :math:`N=100`
     - :math:`176.6 \text{ } (0.87)`
     - :math:`265.4 \text{ } (2.91)`
   * - :math:`N=500`
     - :math:`378.6 \text{ } (48.33)`
     - :math:`703.1 \text{ } (98.49)`
   * - :math:`N=1000`
     - :math:`591.9 \text{ } (31.84)`
     - :math:`1213.6 \text{ } (96.13)`

Code used for timing: See `here <https://github.com/JianhanZhang/CFRL/blob/main/examples/workflow_computing_times/time_real_data_workflow.py>`_.