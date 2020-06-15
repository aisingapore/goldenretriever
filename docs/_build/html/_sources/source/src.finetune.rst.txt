Finetuning GoldenRetriever
====================
GoldenRetriever may be finetuned using the finetuning script. 
The script leverages on triplet generators subpackage 
as well as a triplet loss function. 

Sample usage:

.. highlight:: bash
.. code-block:: bash

    python -m src.finetune.main

Evaluation methods
------------------

.. automodule:: src.finetune.eval
   :members:
   :undoc-members:
   :show-inheritance:

Finetuning Triplet Generators
------------------------------

.. automodule:: src.finetune.generators
   :members:
   :undoc-members:
   :show-inheritance:

