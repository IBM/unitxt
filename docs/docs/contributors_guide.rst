.. _contributors_guide:

==================
Contributors Guide
==================

This guide will assist you in contributing to unitxt.

------------------------
The Unitxt Documentation
------------------------

The unitxt external documentation is at https://unitxt.readthedocs.io/en/main/docs/introduction.html.

The documentation is produced from two sources:

- RST files located within the **docs** directory (https://github.com/IBM/unitxt/tree/main/docs).
- From the docstrings within the library python files. Changes to the docstrings are automatically propagated
  into the documentation for the latest version.

Editing the RST files
*********************

The main file is **index.rst**. Files for the different sections are under **docs/docs**.

To update the documentation, edit the **.rst** documentation files.

To test the documentation locally:

1. Make sure you have the documentation requirements installed:

.. code-block:: console

    pip install -r requirements/docs.rqr

2. Start a local documentation server:

.. code-block:: console

    make docs-server

3. Access the documentation at http://localhost:8478/.

-----------------------------
Creating a new Unitxt release
-----------------------------

The following process describes how to create a new release of Unitxt.

1. In a development environment, checkout the main branch:

.. code-block:: console

    git checkout main

2. Pull the latest code:

.. code-block:: console

    git pull

3. Determine the new version number. Increase the version number
by:

- 1 for major changes (e.g. 1.3.1 -> 2.0)
- 0.1 for regular changes (e.g. 1.3.1 -> 1.4)
- 0.0.1 for small bug fixes or patches (e.g. 1.3.1 -> 1.3.2)

4. Create a branch with the new version number:

.. code-block:: console

    make version=<new version number> new-version

for example:

.. code-block:: console

    make version=1.4.1 new-version

This will create a branch named with the new version number,
and will push the new branch to the remote git Unitxt repo.

5. Create a pull request for merging the new branch to the main branch, on the
Unitxt git repo https://github.com/IBM/unitxt.

6. Squash and merge the new pull request. It is ok to skip the tests for this PR since it changes only the
version number. This can be done by marking "merge without waiting for the requirements" within the
pull request
(note this option may not be available in the UI, since it requires specific permissions that are not given to all contributors).

7. After the merge, pull the merged changes to your local development environment:


.. code-block:: console

    git pull --rebase

Make sure your local main is now after the merge, with an updated version number in **version.py**.

8. Create a new version tag:

.. code-block:: console

    make version-tag

This will tag the main branch with a new tag equal to the updated version number.

9. Go to the Unitxt Releases list: https://github.com/IBM/unitxt/releases.

10. Choose "Draft a new release", and choose the new tag that was just created.
The new release name should be "Unitxt <new version number", for example "Unitxt 1.4.0".

11. Use "Generate release notes" to create an initial list of changed for the new release.
Click "Save Draft" to first save this auto-generated list.

12. Edit the release notes:

- Remove minor items, such as smaller version bumps.
- Add sections Enhancements, Bug fixes, Non backward compatible changes (see release notes of previous versions for examples).
- For each auto-generated item in the "What's Changed" section, copy it, if needed, to one of the above sections.
  Add a description that is concise and clear. Follow previous release notes for examples.

13. Click "Publish release".

14. There are a few actions that are triggered when a new release is published.
The actions are available at https://github.com/IBM/unitxt/actions.

- Check that the action "Release new version to PyPI" completes successfully
  (https://github.com/IBM/unitxt/actions/workflows/pipy.yml).
- The action "Release new version HuggingFace Hub" is currently known to be failing (since 1.2.0).

15. Check that the new release is available on pypi (https://pypi.org/project/unitxt).

