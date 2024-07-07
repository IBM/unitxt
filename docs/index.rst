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
      <a href="https://unitxt.readthedocs.io/en/latest/" class="custom-button">Video</a>
      <a href="https://unitxt.readthedocs.io/en/latest/docs/demo.html" class="custom-button">Demo</a>
      <a href="https://unitxt.readthedocs.io/en/latest/docs/adding_dataset.html" class="custom-button">Tutorial</a>
      <a href="https://arxiv.org/abs/2401.14019" class="custom-button">Paper</a>
      <a href="https://unitxt.readthedocs.io/en/latest/modules.html" class="custom-button">Documentation</a>
      <a href="https://unitxt.readthedocs.io/en/latest/catalog/catalog.__dir__.html" class="custom-button">Catalog</a>
      <a href="https://github.com/IBM/unitxt/blob/main/CONTRIBUTING.md" class="custom-button">Contributors</a>
      <a href="https://pypi.org/project/unitxt/" class="custom-button">PyPi</a>
      <a href="https://unitxt.readthedocs.io/en/latest/search.html" class="custom-button">Search</a>
      <a href="https://unitxt.readthedocs.io/en/latest/py-modindex.html" class="custom-button">Modules</a>
      <br>
   </body>
   </html>

.. raw:: html

   <br>
   <video autoplay muted src="_static/video.mov" width="100%" id="controlled-video">
   </video>


.. raw:: html

   <div>
      <br><br><br><br><br><br><br>
   </div>
   <div class="feed-box-container">
   <div class="feed-box" background-color=rgb(229, 216, 255)>
      <div class="feed-box-text">
         <h1> Loading datasets is easier than ever! </h1>
      </div>
      <div class="feed-code-box" text-align="left">
         <div class="code-window">
            <div class="code-window-buttons">
               <span class="code-window-button close-button"></span>
               <span class="code-window-button minimize-button"></span>
               <span class="code-window-button expand-button"></span>
            </div>
            <div class="code-snippet" data-code="withInstallation" style="display: block;">
               <span><span style="color: #f321e2;">from</span> unitxt <span style="color: #f321e2;">import</span> load_dataset</span>
               <br>
               <br>
               <br>
               <span>dataset <span style="color: #868bff;">=</span> load_dataset<span style="color: #f321e2;">(</span><span style="color: #fdbf02;">"card=cards.sst2"</span><span style="color: #f321e2;">)</span></span>
            </div>
            <div class="code-snippet" data-code="noInstallation" style="display: none;">
               <span><span style="color: #f321e2;">from</span> datasets <span style="color: #f321e2;">import</span> load_dataset</span>
               <br>
               <br>
               <br>
               <span>dataset <span style="color: #868bff;">=</span> load_dataset<span style="color: #f321e2;">(</span><span style="color: #fdbf02;">"unitxt/data"</span>, <span style="color: #fdbf02;">"card=cards.sst2"</span><span style="color: #f321e2;">)</span></span>
            </div>
         </div>
          <div style="text-align: center;">
            <h3 style="display: inline-block; vertical-align: middle; padding: 5px;">No installation mode:</h3>
            <label class="switch">
               <input type="checkbox">
               <span class="slider round"></span>
            </label>
         </div>
      </div>
   </div>
   </div>

   <div>
         <br><br><br><br><br><br><br>
         <h1> Thousands of datasets, templates, prompts and metrics in one place </h1>
         <br><br><br><br><br><br>
         <br><br><br><br><br><br><br>
   </div>

   <div class="feed-box-container">
   <div class="feed-box" background-color=rgb(229, 216, 255)>
      <div class="feed-box-text">
         <h1>Full evaluation in one line of code</h1>
      </div>
      <div class="feed-code-box"  data-box="1" text-align="left">
         <div class="code-window">
            <div class="code-window-buttons">
               <span class="code-window-button close-button"></span>
               <span class="code-window-button minimize-button"></span>
               <span class="code-window-button expand-button"></span>
            </div>
            <div class="code-snippet" data-code="withInstallation" style="display: block;">
               <span><span style="color: #f321e2;">from</span> unitxt <span style="color: #f321e2;">import</span> evaluate</span>
               <br>
               <br>
               <br>
               <span>results <span style="color: #868bff;">=</span>  evaluate<span style="color: #f321e2;">(</span>predictions_list, dataset)</span>
            </div>
            <div class="code-snippet" data-code="noInstallation" style="display: none;">
               <span><span style="color: #f321e2;">from</span> evaluate <span style="color: #f321e2;">import</span> load</span>
               <br>
               <br>
               <br>
               <span>results <span style="color: #868bff;">=</span> load<span style="color: #f321e2;">(</span><span style="color: #fdbf02;">"unitxt/metric"</span><span style="color: #f321e2;">)</span>.compute<span style="color: #f321e2;">(</span>predictions_list, dataset<span style="color: #f321e2;">)</span></span>
            </div>
         </div>
         <br>
         <div style="text-align: center;">
            <h3 style="display: inline-block; vertical-align: middle; padding: 5px;">No installation mode:</h3>
            <label class="switch">
               <input type="checkbox">
               <span class="slider round"></span>
            </label>
         </div>
      </div>
   </div>
   </div>

--------
Welcome!
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   docs/introduction
   docs/demo
   docs/loading_datasets
   docs/evaluating_datasets
   docs/installation
   docs/tutorials
   docs/examples
   docs/data_classification_policy
   docs/rag_support
   docs/operators
   docs/saving_and_loading_from_catalog
   docs/production
   docs/debugging
   docs/helm
   docs/glossary
   documentation
   catalog/catalog.__dir__

