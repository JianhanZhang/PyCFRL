Release Notes
=======================

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