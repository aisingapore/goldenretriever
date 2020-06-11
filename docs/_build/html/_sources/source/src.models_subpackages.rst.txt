GoldenRetriever's Models
===========

GoldenRetriever uses loads ``encoders`` into the main ``src.models.GoldenRetriever`` class.
The ``encoders`` class handles encoding, saving, loading amongst other model specific functions, whereas the ``GoldenRetriever`` class is a wrapper that interfaces with knowledge bases and deployed applications.
It is possible to create a custom encoder to be loaded into the the main ``src.models.GoldenRetriever`` class.

There is also a prebuilt index subpackage that loads encoded knowledge bases into the deployed app. 

Encoders
---------

.. automodule:: src.encoders
   :members: Encoder, USEEncoder
   :undoc-members:
   :show-inheritance:

GoldenRetriever Class
-----------------

.. automodule:: src.models
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
----------------

.. automodule:: src.loss_functions
   :members:
   :undoc-members:
   :show-inheritance:

Prebuilt Index
--------------

.. automodule:: src.prebuilt_index
   :members:
   :undoc-members:
   :show-inheritance:

