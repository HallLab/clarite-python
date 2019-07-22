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
     io
     modify
     plot

.. click:: clarite.cli:analyze_cli
    :prog: clarite-cli analyze
    :show-nested:
.. click:: clarite.cli:describe_cli
    :prog: clarite-cli describe
    :show-nested:
.. click:: clarite.cli:io_cli
    :prog: clarite-cli io
    :show-nested:
.. click:: clarite.cli:modify_cli
    :prog: clarite-cli modify
    :show-nested:
.. click:: clarite.cli:plot_cli
    :prog: clarite-cli plot
    :show-nested:
