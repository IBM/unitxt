.. Unitxt documentation master file, created by
   sphinx-quickstart on Mon Jul 10 03:44:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. raw:: html

   <html>
   <head>
      <style>
         .rounded-image {
               width: 100%;
               height: auto;
               border-top-left-radius: 0.5em;
               border-top-right-radius: 0.5em;
               border-bottom-right-radius: 0.5em;
               border-bottom-left-radius: 0.5em;
         }
      </style>
   </head>
   <body>
      <img src="_static/banner.png" alt="Descriptive Image Text" class="rounded-image">
   </body>
   </html>
   <html>
   <head>
      <style>
         body {
               font-family: Arial, sans-serif;
         }
         .custom-button {
               display: inline-block;
               padding: 0.75em 1em;
               margin: 0.1em;
               border-radius: 0.3em;
               background-color: #fce2ff;
               color: #141216;
               text-align: center;
               text-decoration: none;
               font-weight: bold;
               box-sizing: border-box;
               text-transform: uppercase;
               transition: background-color 0.3s;
         }

         .custom-button:hover {
               background-color: #d1b3d1; /
         }
      </style>
   </head>
   <body>
      <a href="https://www.unitxt.org/en/latest/" class="custom-button">Video</a>
      <a href="https://www.unitxt.org/en/latest/docs/demo.html" class="custom-button">Demo</a>
      <a href="https://www.unitxt.org/en/latest/docs/adding_dataset.html" class="custom-button">Tutorial</a>
      <a href="https://arxiv.org/abs/2401.14019" class="custom-button">Paper</a>
      <a href="https://www.unitxt.org/en/latest/modules.html" class="custom-button">Documentation</a>
      <a href="https://www.unitxt.org/en/latest/catalog.html" class="custom-button">Catalog</a>
      <a href="https://www.unitxt.org/en/latest/docs/contributors_guide.html" class="custom-button">Contributers</a>
      <a href="https://pypi.org/project/unitxt/" class="custom-button">PyPi</a>
      <a href="https://www.unitxt.org/en/latest/search.html" class="custom-button">Search</a>
      <a href="https://www.unitxt.org/en/latest/py-modindex.html" class="custom-button">Modules</a>
      <br>
   </body>
   </html>

.. raw:: html

   <br>
   <video autoplay muted src="_static/video.mov" width="100%" id="controlled-video">
   </video>



Unitxt is a Python library for getting data prepared and ready for utilization in training, evaluation and inference of language models.
It provides a set of reusable building blocks and methodology for defining datasets and metrics.

In one line of code, it prepares a dataset or mixtures-of-datasets into an input-output format for training and evaluation.
Our aspiration is to be simple, adaptable, and transparent.


.. raw:: html

   <div>
   <br><br><br><br><br><br><br>
   </div>
   <div class="feed-box">
   <br><br><br><br><br><br><br>
        <h1> Loading datasets is easier than ever! </h1>
   <br><br><br><br><br><br>
    <img src="_static/loading_code.gif" width="80%" min-width="400px" alt="Descriptive Image Text">
   <br><br><br><br><br><br><br>
   <div class="feed-code-box">
      <pre>
         <code class="language-python" text-align="left">
      from datasets import load_dataset

      dataset = load_dataset("unitxt/data", "card=cards.sst2")
         </code>
      </pre>
   </div>
   </div>
   <div>
   <br><br><br><br><br><br><br>
   <h1> Thousands of datasets, templates, prompts and metrics in one place </h1>
   <br><br><br><br><br><br><br>
   </div>
   <div class="feed-box">
   <br><br><br><br><br><br><br>
        <h1> Full evaluation in one line of code! </h1>
   <br><br><br><br><br><br>
    <img src="_static/evaluating_code.gif" width="80%" min-width="400px" alt="Descriptive Image Text">
   <br><br><br><br><br><br><br>
   </div>

--------
Welcome!
--------

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   docs/introduction
   docs/demo
   docs/no_installation_usage
   docs/installation
   docs/adding_dataset
   docs/adding_operators_and_metrics
   docs/components
   docs/backend
   docs/operators
   docs/contributors_guide
   docs/saving_and_loading_from_catalog
   modules
   catalog

