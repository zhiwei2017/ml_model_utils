ML Model Utils
==============

Introduction
------------
Machine Learning model utils functions.

Prerequisites
-------------
To install the dependencies listed in `requirements/base.txt`, you can use::

    $ pip install -r requirements/base.txt

User Guide
----------

How to Install
++++++++++++++

Stable release
``````````````

To install ML Model Utils, run this command in your terminal:

.. code-block:: console

    $ pip install ml_model_utils

This is the preferred method to install ML Model Utils, as it will always install the most recent stable release.


From sources
````````````

The sources for ML Model Utils can be downloaded from the `Github repo <https://github.com/zhiwei2017/ml_model_utils>`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/zhiwei2017/ml_model_utils.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

or

.. code-block:: console

    $ pip install .

How to Use
++++++++++

Upload Model to mlflow
----------------------

Here is an example for using ml_model_utils to upload model and its metrics to mlflow::

    from ml_model_utils.mlflow import upload_model
    from ml_model_utils.constants import MlflowModelStage

    upload_model("https://mlflow.dummy.com", "dummy-model", "dummy_model",
                 "classifier", model,
                 dict(precision=0.91, recall=0.90, f1_score=0.905, support=300),
                 MlflowModelStage.STAGING, MlflowModelStage.ARCHIVED)


Get files from S3 path
----------------------
Here is an example for gettting files from a given S3 path::

    from ml_model_utils.files import get_s3_files

    get_s3_files("s3://dummy_bucket/dummy/path")

For more usages, please check the section `Source <https://zhiwei2017.github.io/ml_model_utils/02_source.html>`_ from our `documentation <https://zhiwei2017.github.io/ml_model_utils/>`_.

Maintainers
-----------

..
    TODO: List here the people responsible for the development and maintaining of this project.
    Format: **Name** - *Role/Responsibility* - Email

* **Zhiwei Zhang** - *Maintainer* - `zhiwei2017@gmail.com <mailto:zhiwei2017@gmail.com?subject=[GitHub]ML%20Model%20Utils>`_