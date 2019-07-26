=============
CLI Reference
=============

Once CLARITE is installed, the command line interface can be run using the :code:`clarte-cli` command.

The command line interface has command groups that are the same as the modules in the package (except for *survey*).

The :code:`--help` option will show documentation when run with any command or command group:

.. code-block:: bash

   $ clarite-cli --help
   Usage: clarite-cli [OPTIONS] COMMAND [ARGS]...

   Options:
   --help  Show this message and exit.
   
   Commands:
     analyze
     describe
     load
     modify
     plot

--skip and --only
-----------------
Many commands in the CLI have the *skip* and *only* options.  These will limit the command to specific variables.
If *skip* is specified, all variables except the specified ones will be processed.
If *only* is specified, only the specified variables will be processed.

Only one or the other option may be used in a single command.  They may be passed in any combination of two ways:

1. As the name of a file containing one variable name per line
2. As the variable name specfied directly in the terminal

For example:

.. code-block::

   clarite-cli modify rowfilter-incomplete-obs 1_nhanes_w_sddsrvyr test -o covars.txt -o BMXBMI

results in:

.. code-block:: none

   -------------------------------------------------------------------------------------------------------------------------
   
   --only: 1 variable(s) specified directly
           8 variable(s) loaded from 'covars.txt'
   =========================================================================================================================
   
   Running rowfilter_incomplete_obs
   -------------------------------------------------------------------------------------------------------------------------
   
   Removed 3,687 of 22,624 observations (16.30%) due to NA values in any of 9 variables
   =========================================================================================================================

Commands
--------

.. click:: clarite.cli:analyze_cli
    :prog: clarite-cli analyze
    :show-nested:
.. click:: clarite.cli:describe_cli
    :prog: clarite-cli describe
    :show-nested:
.. click:: clarite.cli:load_cli
    :prog: clarite-cli load
    :show-nested:
.. click:: clarite.cli:modify_cli
    :prog: clarite-cli modify
    :show-nested:
.. click:: clarite.cli:plot_cli
    :prog: clarite-cli plot
    :show-nested:
