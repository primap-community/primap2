.. highlight:: shell

=============
Data packages
=============

Individual PRIMAP2 datasets are stored in netcdf files, which preserve all meta
data and the structure of the data.
One or multiple datasets are stored together with the input data and python scripts
needed to generate them in data packages, which are managed with
`datalad <https://www.datalad.org/>`_.
Documentation about datalad can be found in
`its handbook <https://handbook.datalad.org>`_.

Installing datalad
------------------

Datalad depends on multiple components (python, git, and git-annex) and therefore the
installation differs for each platform.
Please refer to the
`datalad handbook <http://handbook.datalad.org/en/latest/intro/installation.html>`_
for detailed installation instructions.

Creating a data package
-----------------------

Detailed information on creating datasets can be found in the
`corresponding section <http://handbook.datalad.org/en/latest/basics/101-101-create.html>`_
in the datalad handbook.
Here, we will show the commands needed to create a dataset for use with PRIMAP2.
To create an empty dataset use the ``datalad create`` command::

    $ datalad create -c text2git <new folder name>

This will create a new folder and populate it with configuration for git, git-annex,
and datalad.
Additionally, it will add configuration such that all text files such as python code
are stored in git with full change tracking, while all binary files such as netcdf files
are added to the annex so that they are transmitted only on demand.
