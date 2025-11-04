---
title: "PyCFRL: A Python library for counterfactually fair offline reinforcement learning via sequential data preprocessing"
tags:
  - counterfactual fairness
  - algorithmic fairness
  - reinforcement learning
  - causal inference
authors:
  - name: Jianhan Zhang
    corresponding: false 
    affiliation: 1
  - name: Jitao Wang
    corresponding: false 
    affiliation: 2
  - name: Chengchun Shi
    corresponding: false 
    affiliation: 3
  - name: John D. Piette
    corresponding: false 
    affiliation: 4
  - name: Donglin Zeng
    corresponding: false 
    affiliation: 2
  - name: Zhenke Wu
    corresponding: true 
    affiliation: 2

affiliations:
 - name: 'Department of Statistics, University of Michigan, USA'
   index: 1
 - name: 'Department of Biostatistics, University of Michigan, USA'
   index: 2
 - name: 'Department of Statistics, London School of Economics, UK'
   index: 3
 - name: 'Department of Health Behavior and Health Equity, School of Public Health, University of Michigan, USA'
   index: 4

citation_author: Zhang et. al.
date: 9 August 2025
year: 2025
journal: JOSS
bibliography: paper.bib
preamble: >
  \usepackage{longtable}
  \usepackage{makecell}
  \usepackage{tabularx}
  \usepackage{hyperref}
  \usepackage{graphicx}
  \usepackage{amsmath}
  \usepackage{booktabs}
  \usepackage{amsfonts}
  \usepackage{tabulary}
  \usepackage{ragged2e}
  \usepackage{floatrow}
  \floatsetup[table]{capposition=top}
  \floatsetup[figure]{capposition=top}
---

# Summary

Reinforcement learning (RL) aims to learn and evaluate a sequential
decision rule, often referred to as a “policy”, that maximizes
expected discounted cumulative rewards to optimize the population-level benefit in an environment across possibly infinitely many time steps. RL has gained popularity in fields such as healthcare, banking, autonomous driving, and, more recently, large language model fine-tuning. However, the sequential decisions made by an RL algorithm, while optimized to maximize overall population benefits, may disadvantage certain individuals who are in minority or socioeconomically disadvantaged groups. A fairness-unaware RL algorithm learns an optimal policy that makes decisions based on the *observed* state variables. However, if certain values of the sensitive attribute influence the state variables and lead the policy to systematically withhold certain actions from an individual, unfairness will result. For example, Hispanics may under-report their pain levels due to cultural factors, misleading a fairness-unaware RL agent to assign less therapist time to these individuals [@piette2023powerED]. Deployment of RL algorithms without careful fairness considerations can raise concerns and erode public trust in high-stakes settings.

To formally define and address the fairness problem in the novel sequential decision-making settings, @wang2025cfrl extended the concept of single-stage counterfactual fairness (CF) in a structural causal framework [@kusner2018cf] to the
multi-stage setting and proposed a data preprocessing algorithm that
ensures CF. A policy is counterfactually fair if, at every time step, the probability of assigning any action does not change had the individual's sensitive attribute taken a different value, while holding constant other historical exogenous variables and actions. In this light, the data preprocessing algorithm ensures CF by constructing new state variables that are not impacted by the sensitive attribute(s). Reward preprocessing is also conducted, but with a different purpose to improve the value of the learned optimal policy rather than to ensure CF. We refer interested readers to @wang2025cfrl for more technical details.

The `PyCFRL` library implements the data preprocessing algorithm proposed by @wang2025cfrl and provides functionalities to evaluate the value (expected discounted cumulative reward) and counterfactual unfairness level achieved by 
any given policy. Here, "CFRL" stands for "Counterfactual Fairness in Reinforcement Learning". The library produces preprocessed trajectories that can be used by
an off-the-shelf offline RL algorithm, such as fitted Q-iteration (FQI) [@riedmiller2005fqi], to learn an optimal CF
policy. The library can also simply read in any policy following a required format and return its
value and counterfactual unfairness level in the environment of interest, where the environment can be either pre-specified or learned from the data.

# Statement of Need

Many existing `Python` libraries implement algorithms designed to ensure fairness
in machine learning. For example, `Fairlearn` [@weerts2023fairlearn] and 
`aif360` [@aif360-oct-2018] provide tools 
for mitigating bias in single-stage machine learning predictions under
statistical association-based fairness criteria such as demographic
parity and equal opportunity. However, existing libraries do not focus on 
counterfactual fairness, which defines an individual-level fairness concept from a causal
perspective, and they cannot be easily extended to the general RL setting. Scripts available from `ml-fairness-gym` [@fairness_gym] allow users 
to simulate unfairness in sequential decision-making, but they neither 
implement algorithms that reduce unfairness nor address CF. To our knowledge, @wang2025cfrl is the first 
work to study CF in RL. 
Correspondingly, `PyCFRL` is also the first code library to address CF in the RL setting.

The contribution of `PyCFRL` is two-fold. First, `PyCFRL` implements a data
preprocessing algorithm that ensures CF in offline RL.
For each individual in the data, the preprocessing
algorithm sequentially estimates and concatenates the counterfactual states under different sensitive
attribute values with the observed state at each time point into a new state vector. The preprocessed data can then be
directly used by existing RL algorithms for policy learning, and the
learned policy will be counterfactually fair up to finite-sample estimation accuracy. Second, `PyCFRL`
provides a platform for assessing RL policies based on CF. After passing in any policy and a data trajectory from the
environment of interest, users can estimate the value and counterfactual unfairness level achieved by the policy in the environment of interest. 

# High-level Design

The `PyCFRL` library is composed of 5 major modules as summarized below.

|Module        |Functionalities                                                                       |
|--------------|--------------------------------------------------------------------------------------|
|`reader`      |Implements functions that read tabular trajectory data into an array format required by `PyCFRL`. Also implements functions that export trajectory data to the tabular format.          |
|`preprocessor`|Implements the data preprocessing algorithm introduced in @wang2025cfrl.              |
|`agents`      |Implements an FQI algorithm [@riedmiller2005fqi], which learns RL policies and makes decisions based on the learned policy.|
|`environment` |Implements a synthetic environment that produces synthetic data as well as a simulated environment that estimates and simulates the transition dynamics of the unknown environment underlying some real-world RL trajectory data. Also implements functions for sampling trajectories from the synthetic and simulated environments.                                        |
|`evaluation`  |Implements functions that evaluate the value and counterfactual unfairness level of a policy.                                  |

A general `PyCFRL` workflow is as follows: First, simulate trajectories using `environment` or read 
in trajectories using `reader`. Then, train a preprocessor using `preprocessor` and preprocess the training trajectory data. After that, pass the preprocessed trajectories into the FQI algorithm in 
`agents` to learn a counterfactually fair policy. Finally, use functions in `evaluation` to 
evaluate the value and counterfactual unfairness level of the trained policy. 

In addition, `PyCFRL` also provides tools to check for potential non-convergence that may arise during the training of neural networks, FQI, or fitted-Q evaluation (FQE). More discussions about non-convergence in `PyCFRL` can be found in the ["Common Issues"](https://pycfrl-documentation.netlify.app/tutorials/common_issues) section of the documentation.

# Data Examples

In the ["Example Workflows"](https://pycfrl-documentation.netlify.app/tutorials/example_workflows) section of the documentation, we provide data examples with code to demonstrate some major workflows of `PyCFRL`. We also record the computing times of different workflows under different combinations of the number of individuals ($N$) and the length of horizons ($T$) in the ["Computing Times"](https://pycfrl-documentation.netlify.app/introduction/computing_times) section of the documentation.

# Conclusions

`PyCFRL` is a `Python` library that enables counterfactually fair reinforcement
learning through data preprocessing. It also provides tools to calculate
the value and unfairness level of a given policy. To our knowledge, it is the first library to address CF
problems in the context of RL. The practical utility of `PyCFRL` can be further improved via extensions. First, the current
`PyCFRL` implementation requires every individual in the offline dataset to
have the same number of time steps. Extending the library to accommodate
variable-length episodes can improve its flexibility and usefulness.
Second, `PyCFRL` can further combine the
preprocessor with popular offline RL algorithm libraries such as
`d3rlpy` [@d3rlpy], or connect the evaluation functions with established RL
environment libraries such as `gym` [@towers2024gymnasium]. Third, generalization to
non-additive counterfactual states reconstruction can make `PyCFRL` more versatile. We leave these extensions 
to future updates.

# Acknowledgements

Jianhan Zhang and Jitao Wang contributed equally to this work. The authors declare no conflicts of interest.

# References
