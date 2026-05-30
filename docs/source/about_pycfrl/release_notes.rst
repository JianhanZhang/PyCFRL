Release Notes
=======================

0.3.2 (released 05/30/2026)
^^^^^^^^^^^^^^^^^^^^^^^

This release updates how the Bellman target is calculated during FQE training and evaluation. 
The previous versions calculates :math:`target = r + \gamma \hat{Q}(s', a_{actual}^{'})`, where :math:`a_{actual}^{'}` is the action actually chosen by the policy and :math:`\hat{Q}` is the estimated Q function from the previous iteration.
This release calculates :math:`target = r + \sum_{a' \in \mathcal{A}} \gamma p(a'|s') \hat{Q}(s', a')`.

Also, the previous version only uses time steps :math:`t \geq 1` for FQE evaluation, while this release now uses all time steps :math:`t \geq 0` for FQE evaluation.

To accomodate the above changes, a new parameter :code:`is_return_probs` is added to the :code:`act()` function in :code:`FQI` and the base class :code:`Agent`.
When :code:`is_return_probs=True`, :code:`act()` will return an array of the probabilities of taking each valid action (instead of returning the actions that are actually taken).
Under this release, all custom agents will need to also include the parameter :code:`is_return_probs` and return the probabilities when :code:`is_return_probs=True`.
Please refer to the corresponding interface documentation for more detailed information about the new parameter and intended code behaviors.

**Note:** To use this new release, code written using earlier versions might need to be updated if they use custom agents.
Existing code that do not use custom agents should still work as intended without modification.

0.3.1 (released 05/27/2026)
^^^^^^^^^^^^^^^^^^^^^^^

This release removes some import statements that import packages not needed 
by the corresponding code.

0.3.0 (released 09/29/2025)
^^^^^^^^^^^^^^^^^^^^^^^

This release updates the importation logic so that classes and functions are 
now accessible as attributes of their containing modules. For example, 
users can now run 

.. code-block:: python

    import pycfrl
    p = pycfrl.environment.SyntheticEnvironment(state_dim=2)

0.2.0 (released 09/29/2025)
^^^^^^^^^^^^^^^^^^^^^^^

This release changes the import path of :code:`PyCFRL`. In previous versions, 
the import path was :code:`import cfrl`. From this version onwards, the 
import path will be :code:`import pycfrl`. This change aligns the import 
path with the library name on PyPI and avoids conflicts. All other 
functionality remains the same.

**Important:** This is a potentially breaking change. We recommend updating 
the import path in existing code to ensure compatibility with this and 
future versions.

0.1.1 (released 09/28/2025)
^^^^^^^^^^^^^^^^^^^^^^^

This release fixes a few typos and mulfunctioning links in the 
:code:`PyCFRL` project page on PyPI. No changes were made to the 
interface or implementation of :code:`PyCFRL`.

0.1.0 (released 09/28/2025)
^^^^^^^^^^^^^^^^^^^^^^^^

This is the initial release of :code:`PyCFRL`.