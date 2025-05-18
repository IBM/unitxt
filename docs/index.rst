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

    .large-text-black {
        font-size: 2em;
        line-height: 1.4;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: normal;
        width: 100%;
        margin: 0 auto;
        color: black;
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
        .container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 20px;
            width: 100%;
            margin: 0 auto;
            max-width: 1200px;
            padding: 20px;
            box-sizing: border-box;
        }

        .button {
            width: 100%;
            min-height: 120px; /* Fixed minimum height */
            background-color: #cfd0ff;
            color: black;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            user-select: none;
            text-align: center;
            padding: 15px;
            font-size: 1rem;
            line-height: 1.4;
            word-wrap: break-word;
            box-sizing: border-box; /* Important to include padding in the size calculation */
            position: relative; /* For proper stacking context */
        }

        /* This makes the anchor tag work like your custom button */
        .button-link {
            text-decoration: none;
            color: inherit;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
            padding: 10px;
            box-sizing: border-box;
        }

        /* Responsive styles for different screen sizes */
        @media (max-width: 1024px) {
            .container {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                grid-gap: 15px;
            }
            .button {
                min-height: 100px;
            }
        }

        .button:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            z-index: 1; /* Ensures the hovering button stays on top */
        }
   </style>
   <div class="container">
      <div class="button">
         <a href="use_cases/existing_data.html" class="button-link">Evaluate models on existing tasks and data</a>
      </div>
      <div class="button">
         <a href="use_cases/user_data.html" class="button-link">Evaluate standard tasks with my data</a>
      </div>
      <div class="button">
         <a href="use_cases/assemble_benchmark.html" class="button-link">Create a Benchmark from Existing Datasets</a>
      </div>
      <div class="button">
         <a href="use_cases/special_tools.html" class="button-link">Evaluate with special processing or metrics</a>
      </div>
      <div class="button">
         <a href="use_cases/llm_as_judges.html" class="button-link">Craft and use LLMs as a Judges</a>
      </div>
      <div class="button">
         <a href="use_cases/multi_modality.html" class="button-link">Evaluate different modalities and data types</a>
      </div>
   </div>

   <script>
      // Modified interaction to prevent overlap issues
      const buttons = document.querySelectorAll('.button');

      buttons.forEach(button => {
         button.addEventListener('mouseenter', () => {
               button.style.transform = 'translateY(-5px)';
               button.style.zIndex = '2'; // Ensure hovered element is on top
         });

         button.addEventListener('mouseleave', () => {
               button.style.transform = 'translateY(0)';
               setTimeout(() => {
                  button.style.zIndex = '1';
               }, 300); // Reset z-index after transition completes
         });

         button.addEventListener('click', () => {
               // Simpler click effect
               button.style.transform = 'translateY(-2px)';
               setTimeout(() => {
                  button.style.transform = 'translateY(-5px)';
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

      <div class="large-text-black">
         Why Unitxt?
      </div>
      <br>
      <br>
      <div class="large-text-black">
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
               color: black;
         }

         .research-text {
               font-size: 4em;
               font-weight: 700; /* Bold weight for "Research" */
               margin: 0;
               padding: 0;
               margin-left: 0.5rem;
               color: black;
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




.. raw:: html

   <style>

        .code-container {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            max-width: 800px;
            margin: 0 auto;
            text-align: left !important; /* Force left alignment */
        }

        .editor-top-bar {
            background-color: #1e1e1e;
            padding: 8px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333;
        }

        .file-tab {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 5px 15px;
            border-radius: 5px 5px 0 0;
            font-size: 13px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .editor-controls {
            display: flex;
            gap: 8px;
        }

        .control-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .dot-red { background-color: #ff5f56; }
        .dot-yellow { background-color: #ffbd2e; }
        .dot-green { background-color: #27c93f; }

        .code-snippet {
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 15px 20px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
            overflow-x: scroll;
            overflow-y: scroll;
            white-space: pre;
            height: 500px;

            /* Force scrollbar appearance */
            scrollbar-width: thin;
            scrollbar-color: #555 #1e1e1e;
        }

        /* WebKit browsers (Chrome, Safari) scrollbar styling */
        .code-snippet::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        .code-snippet::-webkit-scrollbar-track {
            background: #2d2d2d;
        }

        .code-snippet::-webkit-scrollbar-thumb {
            background-color: #555;
            border-radius: 4px;
        }

        .code-snippet::-webkit-scrollbar-thumb:hover {
            background-color: #777;
        }

        .line-numbers {
            color: #858585;
            text-align: right;
            padding-right: 10px;
            user-select: none;
            display: inline-block;
            width: 30px;
        }

        /* Syntax highlighting colors */
        .keyword { color: #569cd6; }
        .function { color: #dcdcaa; }
        .string { color: #ce9178; }
        .comment { color: #6a9955; }
        .variable { color: #9cdcfe; }
        .operator { color: #d4d4d4; }
        .library { color: #4ec9b0; }
        .class { color: #4ec9b0; }
        .number { color: #b5cea8; }
        .constant { color: #4fc1ff; }
        .property { color: #9cdcfe; }
        .punctuation { color: #d4d4d4; }
        .method { color: #dcdcaa; }
        .parameter { color: #9cdcfe; }
        .decorator { color: #c586c0; }
        .bracket { color: #ffd700; }
   </style>
   <br>
   <br>
   <br>
   <br>



   <div class="feed-box-container">
   <div class="feed-box" background-color=rgb(229, 216, 255)>
      <div class="feed-box-text">
         <div class="large-text-black">End to End evaluation made simple</div>
      </div>
         <br>
   <br>
      <div class="code-container">
        <!-- Editor top bar with tabs and controls -->
        <div class="editor-top-bar">
            <div class="file-tab">unitxt_example.py</div>
            <div class="editor-controls">
                <div class="control-dot dot-red"></div>
                <div class="control-dot dot-yellow"></div>
                <div class="control-dot dot-green"></div>
            </div>
        </div>

        <!-- Code with syntax highlighting and line numbers -->
        <div class="code-snippet">
   <span class="line-numbers">1</span> <span class="keyword">from</span> <span class="library">unitxt</span> <span class="keyword">import</span> <span class="function">evaluate</span>, <span class="function">create_dataset</span>
   <span class="line-numbers">2</span> <span class="keyword">from</span> <span class="library">unitxt.blocks</span> <span class="keyword">import</span> <span class="class">Task</span>, <span class="class">InputOutputTemplate</span>
   <span class="line-numbers">3</span> <span class="keyword">from</span> <span class="library">unitxt.inference</span> <span class="keyword">import</span> <span class="class">HFAutoModelInferenceEngine</span>
   <span class="line-numbers">4</span>
   <span class="line-numbers">5</span> <span class="comment"># Question-answer dataset</span>
   <span class="line-numbers">6</span> <span class="variable">data</span> <span class="operator">=</span> [
   <span class="line-numbers">7</span>     {<span class="string">"question"</span>: <span class="string">"What is the capital of Texas?"</span>, <span class="string">"answer"</span>: <span class="string">"Austin"</span>},
   <span class="line-numbers">8</span>     {<span class="string">"question"</span>: <span class="string">"What is the color of the sky?"</span>, <span class="string">"answer"</span>: <span class="string">"Blue"</span>},
   <span class="line-numbers">9</span> ]
   <span class="line-numbers">10</span>
   <span class="line-numbers">11</span> <span class="comment"># Define the task and evaluation metric</span>
   <span class="line-numbers">12</span> <span class="variable">task</span> <span class="operator">=</span> <span class="class">Task</span>(
   <span class="line-numbers">13</span>     <span class="parameter">input_fields</span><span class="operator">=</span>{<span class="string">"question"</span>: <span class="constant">str</span>},
   <span class="line-numbers">14</span>     <span class="parameter">reference_fields</span><span class="operator">=</span>{<span class="string">"answer"</span>: <span class="constant">str</span>},
   <span class="line-numbers">15</span>     <span class="parameter">prediction_type</span><span class="operator">=</span><span class="constant">str</span>,
   <span class="line-numbers">16</span>     <span class="parameter">metrics</span><span class="operator">=</span>[<span class="string">"metrics.accuracy"</span>],
   <span class="line-numbers">17</span> )
   <span class="line-numbers">18</span>
   <span class="line-numbers">19</span> <span class="comment"># Create a template to format inputs and outputs</span>
   <span class="line-numbers">20</span> <span class="variable">template</span> <span class="operator">=</span> <span class="class">InputOutputTemplate</span>(
   <span class="line-numbers">21</span>     <span class="parameter">instruction</span><span class="operator">=</span><span class="string">"Answer the following question."</span>,
   <span class="line-numbers">22</span>     <span class="parameter">input_format</span><span class="operator">=</span><span class="string">"{question}"</span>,
   <span class="line-numbers">23</span>     <span class="parameter">output_format</span><span class="operator">=</span><span class="string">"{answer}"</span>,
   <span class="line-numbers">24</span>     <span class="parameter">postprocessors</span><span class="operator">=</span>[<span class="string">"processors.lower_case"</span>],
   <span class="line-numbers">25</span> )
   <span class="line-numbers">26</span>
   <span class="line-numbers">27</span> <span class="comment"># Prepare the dataset</span>
   <span class="line-numbers">28</span> <span class="variable">dataset</span> <span class="operator">=</span> <span class="function">create_dataset</span>(
   <span class="line-numbers">29</span>     <span class="parameter">task</span><span class="operator">=</span><span class="variable">task</span>,
   <span class="line-numbers">30</span>     <span class="parameter">template</span><span class="operator">=</span><span class="variable">template</span>,
   <span class="line-numbers">31</span>     <span class="parameter">format</span><span class="operator">=</span><span class="string">"formats.chat_api"</span>,
   <span class="line-numbers">32</span>     <span class="parameter">test_set</span><span class="operator">=</span><span class="variable">data</span>,
   <span class="line-numbers">33</span>     <span class="parameter">split</span><span class="operator">=</span><span class="string">"test"</span>,
   <span class="line-numbers">34</span> )
   <span class="line-numbers">35</span>
   <span class="line-numbers">36</span> <span class="comment"># Set up the model (supports Hugging Face, WatsonX, OpenAI, etc.)</span>
   <span class="line-numbers">37</span> <span class="variable">model</span> <span class="operator">=</span> <span class="class">HFAutoModelInferenceEngine</span>(
   <span class="line-numbers">38</span>     <span class="parameter">model_name</span><span class="operator">=</span><span class="string">"Qwen/Qwen1.5-0.5B-Chat"</span>, <span class="parameter">max_new_tokens</span><span class="operator">=</span><span class="number">32</span>
   <span class="line-numbers">39</span> )
   <span class="line-numbers">40</span>
   <span class="line-numbers">41</span> <span class="comment"># Generate predictions and evaluate</span>
   <span class="line-numbers">42</span> <span class="variable">predictions</span> <span class="operator">=</span> <span class="variable">model</span>(<span class="variable">dataset</span>)
   <span class="line-numbers">43</span> <span class="variable">results</span> <span class="operator">=</span> <span class="function">evaluate</span>(<span class="parameter">predictions</span><span class="operator">=</span><span class="variable">predictions</span>, <span class="parameter">data</span><span class="operator">=</span><span class="variable">dataset</span>)
   <span class="line-numbers">44</span>
   <span class="line-numbers">45</span> <span class="comment"># Print results</span>
   <span class="line-numbers">46</span> <span class="function">print</span>(<span class="string">"Global Results:\n"</span>, <span class="variable">results</span>.<span class="property">global_scores</span>.<span class="property">summary</span>)
   <span class="line-numbers">47</span> <span class="function">print</span>(<span class="string">"Instance Results:\n"</span>, <span class="variable">results</span>.<span class="property">instance_scores</span>.<span class="property">summary</span>)
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
   docs/installation
   docs/loading_datasets
   docs/evaluating_datasets
   use_cases/use_cases
   docs/tutorials
   docs/examples
   blog/index
   documentation
   catalog/catalog.__dir__

