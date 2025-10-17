.. PyCFRL documentation master file, created by
   sphinx-quickstart on Sat Jun 14 15:17:14 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyCFRL documentation
==================

Welcome to PyCFRL, a Python library for counterfactually fair reinforcement learning! 
The acronym "CFRL" stands for "Counterfactual Fairness in Reinforcement Learning". PyCFRL provides 
algorithms that ensure counterfactual fairness in reinforcement learning and builds tools for 
evaluating the value and counterfactual fairness of reinforcement learning policies. 

*Note: This library was originally named CFRL, but we later changed the name to PyCFRL.*

To install PyCFRL, run 

.. code-block:: bash
   
   $ pip install pycfrl

This project is still being perfected. We will continue adding new functionalities and expanding 
the use cases of PyCFRL. We appreciate your patience and support!

`[PyCFRL Github repository] <https://github.com/JianhanZhang/CFRL>`_

`[PyCFRL software paper] <https://arxiv.org/abs/2510.06935>`_

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/getting_started
   introduction/computing_times
   introduction/faq

.. toctree::
   :maxdepth: 1
   :caption: Inputs and Outputs

   inputs_and_outputs/data_requirements
   inputs_and_outputs/trajectory_arrays
   inputs_and_outputs/tabular_trajectory_data

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/example_workflows
   tutorials/common_issues

.. toctree::
   :maxdepth: 2
   :caption: Interface

   interface/index

.. toctree::
   :maxdepth: 2
   :caption: Customizations

   customizations/custom_preprocessors
   customizations/custom_agents

.. toctree::
   :maxdepth: 1
   :caption: About PyCFRL

   about_pycfrl/the_pycfrl_team
   about_pycfrl/release_notes