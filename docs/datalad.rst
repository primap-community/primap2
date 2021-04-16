.. highlight:: shell

=============
Data packages
=============

Individual PRIMAP2 datasets are stored in netcdf files, which preserve all meta
data and the structure of the data.
One or multiple datasets are stored together with the input data and python scripts
needed to generate them in data packages, which are managed with
`datalad <https://www.datalad.org/>`_.

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
Here, we will show the commands needed to create a dataset for use with PRIMAP2 at the
PIK infrastructure.
To create an empty dataset use the ``datalad create`` command::

    $ datalad create -c text2git <new folder name>

This will create a new folder and populate it with configuration for git, git-annex,
and datalad.

Store a data package on the cluster
-----------------------------------

Datalad data packages are stored on the PIK file server which can be accessed via
the cluster.
Working with the cluster is easiest if you first configure an ssh alias to access
it without specifying your user name and the host name every time.
For that, put an alias definition into your ``~/.ssh/config``::

    Host pik-cluster
        User <username>
        hostname cluster.pik-potsdam.de

Then you can add the cluster as a remote datalad *sibling* and push your data package
to it::

    $ datalad create-sibling-ria -s origin ria+ssh://pik-cluster/home/pflueger/dlria/
    $ datalad push --to origin

For now, the data package only has a rather unintuitive ID on the cluster - to make
it accessible more easily, add an alias::

    $ # find out the full path of the data package with the datalad siblings command
    $ # example output below
    $ datalad siblings
    .: here(+) [git]
    .: origin-storage(+) [ora]
    .: origin(-) [ssh://pik-cluster/home/pflueger/dlria/1ae/40789-e12c-4c6e-a120-6ebebaade3b7 (git)]
    $ # in this example, the full path is thus
    $ # /home/pflueger/dlria/1ae/40789-e12c-4c6e-a120-6ebebaade3b7
    $ # add a human-readable alias
    $ ssh pik-cluster "ln -s /home/pflueger/dlria/1ae/40789-e12c-4c6e-a120-6ebebaade3b7 /home/pflueger/dlria/alias/<alias name>"

Fetch a data package from the cluster
-------------------------------------

To work on a data package someone else created or if at any time you want to re-fetch
a data package from the cluster, you can clone the data package from the cluster::

    $ datalad clone ria+ssh://pik-cluster/home/pflueger/dlria#~<alias name>

This operation is surprisingly fast even for large data packages.
The reason for this is that by default, only the *metadata* is fetched, i.e. the file
names and descriptions, but not the file contents.
To fetch the file contents, use ``datalad get <PATH>`` to fetch individual files or::

    $ cd <alias name>
    $ datalad get -r .  # translates to "recursively get all content in this directory"

to fetch all data.

Push and pull updates
---------------------

If you made changes to a data package, first save your work::

    $ datalad save -m '<commit message>'

Then push your changes to the cluster using::

    $ datalad push --to origin

To fetch changes made by others from the cluster, use::

    $ datalad update -s origin --merge

For more details check out the section
`"Stay up to date" <http://handbook.datalad.org/en/latest/basics/101-119-sharelocal4.html>`_
in the datalad handbook.
