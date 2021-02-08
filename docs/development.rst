.. highlight:: shell

===========
Development
===========

PRIMAP2 is developed as free software on github, and you are welcome to participate!
To make the development process smooth, we use a couple of tools and standards also
known from other Python projects. In this section, we will describe these tools as well
as the internal structure of the PRIMAP2 library to help you get started.

Quickstart
----------

Here's how to set up a local environment for `primap2` development.

1. **Fork the `primap2` repo on GitHub.**
   Go to `the primap2 github repo <https://github.com/pik-primap/primap2>`_, log into
   github and press the "Fork" button in the top right corner.

2. **Clone your fork locally.**
   If you use an IDE like `pycharm <https://www.jetbrains.com/de-de/pycharm/>`_ or
   `Visual Studio Code <https://code.visualstudio.com/>`_, use your IDE to check out
   your fork from github. For pycharm, use ``git -> Clone…`` or
   ``VCS -> Check out from version control``. Alternatively, if you are not using an
   IDE, clone using the git command line::

    $ git clone git@github.com:your_name_here/primap2.git

3. **Create a virtual environment.**
   To separate your environment used for developing PRIMAP2 from your system python,
   create a virtual environment. After cloning, pycharm will automatically offer
   creating a virtual environment, just accept. Alternatively, you can use the command
   line::

    $ cd primap2/
    $ make virtual-environment

4. **Install pre-commit hooks.**
   For static analysis tools and for enforcing a common code style, we use git hooks
   which are automatically executed before every commit. To install them for yourself,
   execute in the command line::

    $ make install-pre-commit

   You can immediately try the checks using::

    $ make lint

5. **Create a branch.**
   It is best practice to do all of your development in git branches, which you can
   easily submit fo inclusion into PRIMAP2 later. In pycharm, open the git tab,
   right-click on the local ``main`` branch and select ``New branch from selected…``.
   Alternatively, you can use the git command line::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. **Run tests and format.**
   Whenever you feel like it during development, and especially before committing, you
   can run the test suite of PRIMAP2. It consists of two parts: static analysis and
   formatting (called linting) and units tests. To run the static analysis and format
   your code, run::

    $ make lint

   If static analysis finds an error or an inconsistency, this will be highlighted in
   red in the terminal. If the formatting changes your code, it will also be highlighted
   in red. If you are unsure if any action from your side is necessary to pass linting,
   just run ``make lint`` twice in a row. If the second run highlights anything in red,
   you have to fix it yourself.

   To run the unit tests, run in the terminal::

    $ make test

   or use the function provided by your IDE for pytest. If any tests fail, this will
   be shown.

7. **Commit your changes.**
   To commit your changes using pycharm, select the `Commit` tab, select the changed
   files, provide a commit message and use "Commit and Push…". To do the same in the
   terminal, run::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

   Note that the commit will fail if your code does not pass ``make lint``. You have
   to fix all issues listed by ``make lint`` before committing. You can try committing
   twice in a row to see if the linting could fix all issues by itself or if you have
   to fix something yourself.

8. **Submit a pull request.**
   Visit your `primap2` fork on GitHub and submit your branch as a pull request.

That's it! For more details for each particular topic, keep reading.

Branches and Pull Requests
--------------------------

Code format
-----------

Linting
-------

Repo structure
--------------

Adding new functions
--------------------

Testing
-------

Documentation
-------------

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.rst).
Then run::

    $ bump2version patch # possible: major / minor / patch
    $ git push
    $ git push --tags
