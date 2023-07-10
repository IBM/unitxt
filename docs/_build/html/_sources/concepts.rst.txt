==============
Concepts
==============

The main building blocks of the library are the `Stream`, `Operator` and the `Atifact` classes.

Stream
-------

A stream is a sequence of data. It can be finite or infinite. It can be synchronous or asynchronous.
Every instance in the stream is a simple python dictionary.

Operator
---------

An operator is a class that takes multiple streams as input and produces multiple streams as output.
Every modification of the data in the stream is done by an operator.
Every opertor should be doing a single task and its name should reflect its operation.

Examples: AddDictToEveryInstance, RenameField, etc.

Artifact
---------

An artifact is a class that can be saved in human readable format. 
Then it can be edited in the text editor and shared between different projects.
Every operator or pipeline of operators should be saved as an artifact.

Recipe
-------
A data prepration recipe is consisted of a sequence of operators.
The recipe can be easily understood by looking at the list of operations its consisted of. 
The recipe is saved as an artifact and can be shared between different projects
and allow reproducible and transparent data preparation.
