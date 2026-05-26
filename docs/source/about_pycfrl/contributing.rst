Contributing
==================

Contrubuting to :code:`PyCFRL` includes, but is not limited to, reporting bugs 
and other issues, suggesting new features or improvements, and participating 
in discussions of issues and possible improvements. Such contributions are crucial 
to :code:`PyCFRL`'s continual development and are highly appreciated.

In general, contributions to :code:`PyCFRL` should be made on GitHub through 
issues and pull requests. Issues are used for raising questions and discussions, 
while pull requests are used for suggesting changes to the code. We specify below 
guidelines for some common types of contributions.

Reporting and Fixing Bugs
----------------------------

To report a bug, please first search existing issues for the bug. If the bug is not 
yet reported, please open an issue containing the following information:

1. A title that starts with [BUG].
2. A description of the bug. Notably, please include the both expected behavior of the code and the actual, incorrect behavior of the current code.
3. A piece of code which, when executed, demonstrates the incorrect behavior of the current code.
4. Any other information that might be helpful.

In addition, please make a separate issue for each bug you would like to report.

You are welcome to suggest fixes to the bugs reported by you or others. To do so, 
please send a pull request with proposed code changes following the guidelines 
below:

1. Before sending the pull request to suggest bug fixes, please first open an issue to report the bugs if the bugs are not yet reported. It is OK to fix multiple bugs in one pull request. Please reference all the corresponding issues in the pull request.
2. Before sending the pull request to suggest bug fixes, please ensure all the unit tests pass. The unit tests can be run using :code:`python -m pytest`.
3. If needed, please update the documentation (including the docstrings) to reflect the code changes, and include the documentation updates in the pull request.

Suggesting New Features or Improvements
----------------------------

To suggest new features or other improvements (e.g. code optimization), please first 
search existing issues. If the features or improvements have not been proposed before, 
please open an issue containing the following information:

1. A title that starts with [NEW FEATURE] (for suggesting new features) or [IMPROVEMENT] (for suggesting other improvements).
2. A description of the feature or improvement suggested.
3. A statement of how incorporating the suggested feature or improvement to :code:`PyCFRL` would benefit the community.
4. Any other information that might be helpful.

In addition, please make a separate issue for each feature or improvement you would 
like to report.

If a maintainer agrees with the suggested feature or improvement, you are highly welcome 
(but not required) to send pull requests that incorporates the suggested feature or 
improvement into :code:`PyCFRL`. You are also welcome to send pull requests the incorporates 
features or improvements suggested by others. If you kindly would like to send such pull 
requests, please follow the guidelines below:

1. Before working on the pull request, please make sure that a maintainer has agreed with the suggested features or improvements to avoid wasting time on pull requests that we are unable to accept. It is OK to implement multiple features and/or other improvements in one pull request. In the pull request, please reference the issues corresponding to each of the features and/or other improvements being implemented by the pull request.
2. If needed, please add additional unit tests to :code:`PyCFRL/tests` and make sure these unit tests pass. Please also make sure all existing unit tests pass before sending the pull request. 
3. If needed, please update the documentation (including the docstrings) to reflect the changes, and include the documentation updates in the pull request.

Reporting and Fixing Errors in Documentation
----------------------------

To report a documentation error, please first search existing issues for the error. If 
the error is not yet reported, please open an issue containing the following information:

1. A title that starts with [DOC ERROR].
2. A description of the error. 
3. Any other information that might be helpful.

In addition, please make a separate issue for each error you would like to report.

You are welcome to suggest fixes to the errors reported by you or others. To do so, 
please first open an issue to report the errors if the errors are not yet reported. 
After that, please send a pull request with proposed changes made to :code:`PyCFRL/docs`. 
It is OK to fix multiple errors in one pull request. Please reference all the corresponding 
issues in the pull request.

The documentation is created using :code:`Sphinx`, so some familiarity with :code:`Sphinx` 
might be helpful for suggesting fixes to documentation errors. Please see 
`here <https://www.sphinx-doc.org/en/master/>`_ for more information regarding 
:code:`Sphinx`.

Any Other Questions?
----------------------------

For any questions not explained above, and for any contributions that cannot be made 
through issues or pull requests, please contact the maintainer at [jianhanz@umich.edu].
Thanks for making :code:`PyCFRL` better!