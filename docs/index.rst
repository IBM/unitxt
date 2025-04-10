.. raw:: html

   <br>
   <br>
   <div style="text-align: center;">
      <img src="_static/banner.png" alt="Descriptive Image Text" style="width: 80%;">
   </div>


.. raw:: html


   <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700&display=swap');

    .large-text {
        font-size: 2em;
        line-height: 1.4;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: normal;
        width: 100%;
        margin: 0 auto;
    }
   </style>

   <div style="text-align: center; margin: 2em 0;">
      <br><br>
      <div class="large-text">
         Unitxt is a Python library for enterprise-grade evaluation of AI performance, offering the world's largest catalog of tools and data for end-to-end AI benchmarking.
      </div>
   </div>

.. raw:: html

   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <div style="text-align: center; margin: 2em 0;">
   <br><br>
      <div class="large-text">
         How would you like to start?
      </div>
   </div>
   <br>
   <br>
   <style>
        body {

        .container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 20px;
            width: 100%;
            margin: 0 auto;
        }

        .button {
            width: 100%;
            height: 80px;
            background-color: #4a87ff;
            color: black;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.2em;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            user-select: none;
            text-align: center;
            word-wrap: break-word;
        }

        /* Responsive styles for mobile devices */
        @media (max-width: 768px) {
            .container {
                width: 100%;
                grid-template-columns: 1fr;
            }
        }

        .button:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }

        .button:nth-child(1) { background-color: #cfd0ff; }
        .button:nth-child(2) { background-color: #cfd0ff; }
        .button:nth-child(3) { background-color: #cfd0ff; }
        .button:nth-child(4) { background-color: #cfd0ff; }
        .button:nth-child(5) { background-color: #cfd0ff; }
        .button:nth-child(6) { background-color: #cfd0ff; }
    </style>

    <div class="container">
        <div class="button">Evaluate models on existing tasks and data</div>
        <div class="button">Evaluate standard tasks with my data</div>
        <div class="button">Create a Benchmark from Existing Datasets</div>
        <div class="button">Evaluate with special processing or metrics</div>
        <div class="button">Craft and use LLMs as a Judges</div>
        <div class="button">Evaluate different modalities and data types</div>
    </div>

    <script>
        // Add a little extra interaction
        const buttons = document.querySelectorAll('.button');

        buttons.forEach(button => {
            button.addEventListener('mouseenter', () => {
                button.style.transform = 'translateY(-5px) scale(1.05)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translateY(0) scale(1)';
            });

            button.addEventListener('click', () => {
                // Add a quick scale effect on click
                button.style.transform = 'translateY(-5px) scale(0.95)';
                setTimeout(() => {
                    button.style.transform = 'translateY(-5px) scale(1.05)';
                }, 100);
            });
        });
    </script>

.. raw:: html

   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <div class="feed-box-container">
   <div class="feed-box" background-color=rgb(229, 216, 255)>
   <div style="text-align: center; margin: 2em 0;">
   <br><br><br><br>
      <div class="large-text">
         Why Unitxt?
      </div>
      <br>
      <br>
      <div class="large-text">
         Unitxt was built by IBM Research to host a maintainable large collection of evaluation assets. If you care about robust evaluation that last, then Unitxt is for you.
      </div>
      <br>
      <br>
      <style>
         @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700&display=swap');

         .ibm-logo {
               display: flex;
               align-items: baseline;
               font-family: 'IBM Plex Sans', sans-serif;
               justify-content: center; /* Centers horizontally in a flex container */
               margin: 0 auto; /* Centers the block if it has a width */
               width: 100%; /* Makes the container take full width */
         }

         .ibm-text {
               font-size: 4em;
               font-weight: 400;
               letter-spacing: -0.5px;
               margin: 0;
               padding: 0;
         }

         .research-text {
               font-size: 4em;
               font-weight: 700; /* Bold weight for "Research" */
               margin: 0;
               padding: 0;
               margin-left: 0.5rem;
         }
      </style>
      <div class="ibm-logo">
        <div class="ibm-text">IBM</div>
        <div class="research-text">Research</div>
      </div>
   </div>
   </div>
   </div>
      <br>
   <br>
   <br>

.. raw:: html

   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <div class="large-text">
         Unitxt catalog is a one stop shop containing well documented assets constructing robust evaluation pipelines such as: Task Instructions, Data Loaders, Data Types Serializers and Inference Engines
   </div>
   <br>
   <br>
   <style>
   .asset-container {
   text-align: center;
   padding: 20px;
   font-family: Arial, sans-serif;
   }

   .bubbles-wrapper {
   display: flex;
   justify-content: center;
   align-items: center;
   flex-wrap: wrap;
   gap: 15px;
   width: 120%;
   margin: 0 auto;
   transform: translateX(-10%); /* Centers the oversized container */
   overflow: visible;
   }

   .bubble {
   background-color: #3B82F6;
   border-radius: 50%;
   display: flex;
   flex-direction: column;
   justify-content: center;
   align-items: center;
   color: white;
   text-align: center;
   box-shadow: 0 2px 4px rgba(0,0,0,0.2);
   position: relative;
   animation-duration: 3s;
   animation-iteration-count: infinite;
   animation-timing-function: ease-in-out;
   }

   .bubble-1 {
   animation-name: float1;
   }

   .bubble-2 {
   animation-name: float2;
   animation-delay: 0.5s;
   }

   .bubble-3 {
   animation-name: float3;
   animation-delay: 0.2s;
   }

   .bubble-4 {
   animation-name: float4;
   animation-delay: 0.7s;
   }

   .bubble-5 {
   animation-name: float5;
   animation-delay: 0.3s;
   }

   @keyframes float1 {
   0% { transform: translateY(0px); }
   50% { transform: translateY(-5px); }
   100% { transform: translateY(0px); }
   }

   @keyframes float2 {
   0% { transform: translateY(0px); }
   50% { transform: translateY(-7px); }
   100% { transform: translateY(0px); }
   }

   @keyframes float3 {
   0% { transform: translateY(0px); }
   50% { transform: translateY(-4px); }
   100% { transform: translateY(0px); }
   }

   @keyframes float4 {
   0% { transform: translateY(0px); }
   50% { transform: translateY(-6px); }
   100% { transform: translateY(0px); }
   }

   @keyframes float5 {
   0% { transform: translateY(0px); }
   50% { transform: translateY(-3px); }
   100% { transform: translateY(0px); }
   }

   .bubble-name {
   font-size: 13px;
   margin-bottom: 4px;
   padding: 0 5px;
   }

   .bubble-count {
   font-size: 15px;
   font-weight: bold;
   }

   .legend {
   font-size: 12px;
   color: #666;
   margin-top: 10px;
   }
   </style>

   <div class="asset-container">

   <div style="overflow: hidden; width: 100%; position: relative;">
      <div class="bubbles-wrapper">
      <!-- Evaluation Tasks -->
      <div class="bubble bubble-1" style="width: 100px; height: 100px;">
         <div class="bubble-name">Evaluation Tasks</div>
         <div class="bubble-count">64</div>
      </div>

      <!-- LLM Ready Datasets -->
      <div class="bubble bubble-2" style="width: 180px; height: 180px;">
         <div class="bubble-name">LLM Ready<br>Datasets</div>
         <div class="bubble-count">3,174</div>
      </div>

      <!-- Prompts -->
      <div class="bubble bubble-3" style="width: 130px; height: 130px;">
         <div class="bubble-name">Prompts</div>
         <div class="bubble-count">342</div>
      </div>

      <!-- Metrics -->
      <div class="bubble bubble-4" style="width: 140px; height: 140px;">
         <div class="bubble-name">Metrics</div>
         <div class="bubble-count">462</div>
      </div>

      <!-- Custom Benchmarks -->
      <div class="bubble bubble-5" style="width: 80px; height: 80px;">
         <div class="bubble-name">Custom<br>Benchmarks</div>
         <div class="bubble-count">6</div>
      </div>
      </div>
   </div>

   </div>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>



.. raw:: html

   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <div class="feed-box-container">
   <div class="feed-box" background-color=rgb(229, 216, 255)>
      <div class="feed-box-text">
         <h1>End to End evaluation made simple</h1>
      </div>
      <div class="feed-code-box"  data-box="1" text-align="left">
         <div class="code-window">
            <div class="code-window-buttons">
               <span class="code-window-button close-button"></span>
               <span class="code-window-button minimize-button"></span>
               <span class="code-window-button expand-button"></span>
            </div>
            <div class="code-snippet" style="display: block;">
               <span><span style="color: #f321e2;">from</span> unitxt <span style="color: #f321e2;">import</span> evaluate, create_dataset</span>
               <br>
               <span><span style="color: #f321e2;">from</span> unitxt.blocks <span style="color: #f321e2;">import</span> Task, InputOutputTemplate</span>
               <br>
               <span><span style="color: #f321e2;">from</span> unitxt.inference <span style="color: #f321e2;">import</span> HFAutoModelInferenceEngine</span>
               <br>
               <span><span style="color: #868bff;"># Question-answer dataset</span></span>
               <br>
               <span>data <span style="color: #868bff;">=</span> [</span>
               <br>
               <span>    {"question": "What is the capital of Texas?", "answer": "Austin"},</span>
               <br>
               <span>    {"question": "What is the color of the sky?", "answer": "Blue"},</span>
               <br>
               <span>]</span>
               <br>
               <span><span style="color: #868bff;"># Define the task and evaluation metric</span></span>
               <br>
               <span>task <span style="color: #868bff;">=</span> Task(</span>
               <br>
               <span>    input_fields={"question": str},</span>
               <br>
               <span>    reference_fields={"answer": str},</span>
               <br>
               <span>    prediction_type=str,</span>
               <br>
               <span>    metrics=["metrics.accuracy"],</span>
               <br>
               <span>)</span>
               <br>
               <span><span style="color: #868bff;"># Create a template to format inputs and outputs</span></span>
               <br>
               <span>template <span style="color: #868bff;">=</span> InputOutputTemplate(</span>
               <br>
               <span>    instruction="Answer the following question.",</span>
               <br>
               <span>    input_format="{question}",</span>
               <br>
               <span>    output_format="{answer}",</span>
               <br>
               <span>    postprocessors=["processors.lower_case"],</span>
               <br>
               <span>)</span>
               <br>
               <span><span style="color: #868bff;"># Prepare the dataset</span></span>
               <br>
               <span>dataset <span style="color: #868bff;">=</span> create_dataset(</span>
               <br>
               <span>    task=task,</span>
               <br>
               <span>    template=template,</span>
               <br>
               <span>    format="formats.chat_api",</span>
               <br>
               <span>    test_set=data,</span>
               <br>
               <span>    split="test",</span>
               <br>
               <span>)</span>
               <br>
               <span><span style="color: #868bff;"># Set up the model (supports Hugging Face, WatsonX, OpenAI, etc.)</span></span>
               <br>
               <span>model <span style="color: #868bff;">=</span> HFAutoModelInferenceEngine(</span>
               <br>
               <span>    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32</span>
               <br>
               <span>)</span>
               <br>
               <span><span style="color: #868bff;"># Generate predictions and evaluate</span></span>
               <br>
               <span>predictions <span style="color: #868bff;">=</span> model(dataset)</span>
               <br>
               <span>results <span style="color: #868bff;">=</span> evaluate(predictions=predictions, data=dataset)</span>
               <br>
               <span><span style="color: #868bff;"># Print results</span></span>
               <br>
               <span>print("Global Results:\n", results.global_scores.summary)</span>
               <br>
               <span>print("Instance Results:\n", results.instance_scores.summary)</span>
            </div>
         </div>
         <br>

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
   docs/installation
   docs/loading_datasets
   docs/evaluating_datasets
   docs/tutorials
   docs/examples
   blog/index
   documentation
   catalog/catalog.__dir__

