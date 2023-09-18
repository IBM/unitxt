.. _catalog:

----------

Catalog
=======



.. _.formats:

----------

formats
-------



.. _formats.user_agent:

----------

user_agent
^^^^^^^^^^

.. note:: ID: ``formats.user_agent``  |  Type: :class:`ICLFormat <unitxt.formats.ICLFormat>`

   .. code-block:: json

      {
          "input_prefix": "User:",
          "output_prefix": "Agent:",
          "type": "icl_format"
      }


|
|



.. _.templates:

----------

templates
---------



.. _.templates.qa:

----------

qa
^^



.. _.templates.qa.open:

----------

open
""""



.. _templates.qa.open.simple:

----------

simple
''''''

.. note:: ID: ``templates.qa.open.simple``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Question: {question}",
          "output_format": "{answer}",
          "type": "input_output_template"
      }


|
|



.. _templates.qa.open.simple2:

----------

simple2
'''''''

.. note:: ID: ``templates.qa.open.simple2``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "answer the question: {question}",
          "output_format": "{answer}",
          "type": "input_output_template"
      }


|
|



.. _templates.qa.open.all:

----------

all
'''

.. note:: ID: ``templates.qa.open.all``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              "templates.qa.open.simple",
              "templates.qa.open.simple2"
          ],
          "type": "templates_list"
      }

References: :ref:`templates.qa.open.simple2 <templates.qa.open.simple2>`, :ref:`templates.qa.open.simple <templates.qa.open.simple>`

|
|



.. _.templates.qa.contextual:

----------

contextual
""""""""""



.. _templates.qa.contextual.simple:

----------

simple
''''''

.. note:: ID: ``templates.qa.contextual.simple``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Context: {context}\nQuestion: {question}",
          "output_format": "{answer}",
          "type": "input_output_template"
      }


|
|



.. _templates.qa.contextual.simple2:

----------

simple2
'''''''

.. note:: ID: ``templates.qa.contextual.simple2``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "based on this text: {context}\n answer the question: {question}",
          "output_format": "{answer}",
          "type": "input_output_template"
      }


|
|



.. _templates.qa.contextual.all:

----------

all
'''

.. note:: ID: ``templates.qa.contextual.all``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              "templates.qa.contextual.simple",
              "templates.qa.contextual.simple2"
          ],
          "type": "templates_list"
      }

References: :ref:`templates.qa.contextual.simple2 <templates.qa.contextual.simple2>`, :ref:`templates.qa.contextual.simple <templates.qa.contextual.simple>`

|
|



.. _.templates.translation:

----------

translation
^^^^^^^^^^^



.. _.templates.translation.directed:

----------

directed
""""""""



.. _templates.translation.directed.simple:

----------

simple
''''''

.. note:: ID: ``templates.translation.directed.simple``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Translate from {source_language} to {target_language}: {text}",
          "output_format": "{translation}",
          "type": "input_output_template"
      }


|
|



.. _templates.translation.directed.all:

----------

all
'''

.. note:: ID: ``templates.translation.directed.all``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              "templates.translation.directed.simple"
          ],
          "type": "templates_list"
      }

References: :ref:`templates.translation.directed.simple <templates.translation.directed.simple>`

|
|



.. _templates.nli:

----------

nli
^^^

.. note:: ID: ``templates.nli``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              {
                  "input_format": "Given this sentence: {premise}, classify if this sentence: {hypothesis} is {choices}.",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          ],
          "type": "templates_list"
      }


|
|



.. _.templates.mmlu:

----------

mmlu
^^^^



.. _templates.mmlu.fm_eval:

----------

fm_eval
"""""""

.. note:: ID: ``templates.mmlu.fm_eval``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.mmlu.lm_eval_harness:

----------

lm_eval_harness
"""""""""""""""

.. note:: ID: ``templates.mmlu.lm_eval_harness``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.mmlu.original:

----------

original
""""""""

.. note:: ID: ``templates.mmlu.original``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.mmlu.helm:

----------

helm
""""

.. note:: ID: ``templates.mmlu.helm``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _.templates.classification:

----------

classification
^^^^^^^^^^^^^^



.. _.templates.classification.choices:

----------

choices
"""""""



.. _templates.classification.choices.simple:

----------

simple
''''''

.. note:: ID: ``templates.classification.choices.simple``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Text: {text}, Choices: {choices}.",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.classification.choices.simple2:

----------

simple2
'''''''

.. note:: ID: ``templates.classification.choices.simple2``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Given this sentence: {sentence}, classify if it is {choices}.",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.classification.choices.all:

----------

all
'''

.. note:: ID: ``templates.classification.choices.all``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              "templates.classification.choices.simple",
              "templates.classification.choices.simple2",
              "templates.classification.choices.informed"
          ],
          "type": "templates_list"
      }

References: :ref:`templates.classification.choices.informed <templates.classification.choices.informed>`, :ref:`templates.classification.choices.simple <templates.classification.choices.simple>`, :ref:`templates.classification.choices.simple2 <templates.classification.choices.simple2>`

|
|



.. _templates.classification.choices.informed:

----------

informed
''''''''

.. note:: ID: ``templates.classification.choices.informed``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Classify the follwoing text to one of the options: {choices}, Text: {text}",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _.templates.classification.nli:

----------

nli
"""



.. _templates.classification.nli.simple:

----------

simple
''''''

.. note:: ID: ``templates.classification.nli.simple``  |  Type: :class:`InputOutputTemplate <unitxt.templates.InputOutputTemplate>`

   .. code-block:: json

      {
          "input_format": "Given this sentence: {premise}, classify if this sentence: {hypothesis} is {choices}.",
          "output_format": "{label}",
          "type": "input_output_template"
      }


|
|



.. _templates.classification.nli.all:

----------

all
'''

.. note:: ID: ``templates.classification.nli.all``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              "templates.classification.nli.simple"
          ],
          "type": "templates_list"
      }

References: :ref:`templates.classification.nli.simple <templates.classification.nli.simple>`

|
|



.. _templates.one_sent_classification:

----------

one_sent_classification
^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``templates.one_sent_classification``  |  Type: :class:`TemplatesList <unitxt.templates.TemplatesList>`

   .. code-block:: json

      {
          "items": [
              {
                  "input_format": "Given this sentence: {sentence}, classify if it is {choices}.",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          ],
          "type": "templates_list"
      }


|
|



.. _.metrics:

----------

metrics
-------



.. _metrics.wer:

----------

wer
^^^

.. note:: ID: ``metrics.wer``  |  Type: :class:`Wer <unitxt.metrics.Wer>`

   .. code-block:: json

      {
          "type": "wer"
      }


|
|



.. _metrics.spearman:

----------

spearman
^^^^^^^^

.. note:: ID: ``metrics.spearman``  |  Type: :class:`MetricPipeline <unitxt.metrics.MetricPipeline>`

   .. code-block:: json

      {
          "main_score": "spearmanr",
          "metric": {
              "main_score": "spearmanr",
              "metric_name": "spearmanr",
              "type": "huggingface_metric"
          },
          "preprocess_steps": [
              {
                  "field_to_field": [
                      [
                          "references/0",
                          "references"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "failure_defaults": {
                      "prediction": 0.0
                  },
                  "fields": {
                      "prediction": "float",
                      "references": "float"
                  },
                  "type": "cast_fields",
                  "use_nested_query": true
              }
          ],
          "type": "metric_pipeline"
      }


|
|



.. _metrics.matthews_correlation:

----------

matthews_correlation
^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``metrics.matthews_correlation``  |  Type: :class:`MatthewsCorrelation <unitxt.metrics.MatthewsCorrelation>`

   .. code-block:: json

      {
          "type": "matthews_correlation"
      }


|
|



.. _metrics.bleu:

----------

bleu
^^^^

.. note:: ID: ``metrics.bleu``  |  Type: :class:`Bleu <unitxt.metrics.Bleu>`

   .. code-block:: json

      {
          "type": "bleu"
      }


|
|



.. _metrics.ner:

----------

ner
^^^

.. note:: ID: ``metrics.ner``  |  Type: :class:`NER <unitxt.metrics.NER>`

   .. code-block:: json

      {
          "type": "ner"
      }


|
|



.. _metrics.rouge:

----------

rouge
^^^^^

.. note:: ID: ``metrics.rouge``  |  Type: :class:`Rouge <unitxt.metrics.Rouge>`

   .. code-block:: json

      {
          "type": "rouge"
      }


|
|



.. _metrics.f1_macro_multi_label:

----------

f1_macro_multi_label
^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``metrics.f1_macro_multi_label``  |  Type: :class:`F1MacroMultiLabel <unitxt.metrics.F1MacroMultiLabel>`

   .. code-block:: json

      {
          "type": "f1_macro_multi_label"
      }


|
|



.. _metrics.char_edit_dist_accuracy:

----------

char_edit_dist_accuracy
^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``metrics.char_edit_dist_accuracy``  |  Type: :class:`CharEditDistanceAccuracy <unitxt.metrics.CharEditDistanceAccuracy>`

   .. code-block:: json

      {
          "type": "char_edit_distance_accuracy"
      }


|
|



.. _metrics.f1_macro:

----------

f1_macro
^^^^^^^^

.. note:: ID: ``metrics.f1_macro``  |  Type: :class:`F1Macro <unitxt.metrics.F1Macro>`

   .. code-block:: json

      {
          "type": "f1_macro"
      }


|
|



.. _metrics.accuracy:

----------

accuracy
^^^^^^^^

.. note:: ID: ``metrics.accuracy``  |  Type: :class:`Accuracy <unitxt.metrics.Accuracy>`

   .. code-block:: json

      {
          "type": "accuracy"
      }


|
|



.. _metrics.f1_micro:

----------

f1_micro
^^^^^^^^

.. note:: ID: ``metrics.f1_micro``  |  Type: :class:`F1Micro <unitxt.metrics.F1Micro>`

   .. code-block:: json

      {
          "type": "f1_micro"
      }


|
|



.. _metrics.f1_micro_multi_label:

----------

f1_micro_multi_label
^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``metrics.f1_micro_multi_label``  |  Type: :class:`F1MicroMultiLabel <unitxt.metrics.F1MicroMultiLabel>`

   .. code-block:: json

      {
          "type": "f1_micro_multi_label"
      }


|
|



.. _metrics.squad:

----------

squad
^^^^^

.. note:: ID: ``metrics.squad``  |  Type: :class:`MetricPipeline <unitxt.metrics.MetricPipeline>`

   .. code-block:: json

      {
          "main_score": "f1",
          "metric": {
              "main_score": "f1",
              "metric_name": "squad",
              "scale": 100.0,
              "type": "huggingface_metric"
          },
          "preprocess_steps": [
              {
                  "type": "add_id"
              },
              {
                  "fields": {
                      "prediction_template": {
                          "id": "ID",
                          "prediction_text": "PRED"
                      },
                      "reference_template": {
                          "answers": {
                              "answer_start": [
                                  -1
                              ],
                              "text": "REF"
                          },
                          "id": "ID"
                      }
                  },
                  "type": "add_fields",
                  "use_deepcopy": true
              },
              {
                  "field_to_field": [
                      [
                          "references",
                          "reference_template/answers/text"
                      ],
                      [
                          "prediction",
                          "prediction_template/prediction_text"
                      ],
                      [
                          "id",
                          "prediction_template/id"
                      ],
                      [
                          "id",
                          "reference_template/id"
                      ],
                      [
                          "reference_template",
                          "references"
                      ],
                      [
                          "prediction_template",
                          "prediction"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              }
          ],
          "type": "metric_pipeline"
      }


|
|



.. _.cards:

----------

cards
-----



.. _cards.race_all:

----------

race_all
^^^^^^^^

.. note:: ID: ``cards.race_all``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "all",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.race_middle:

----------

race_middle
^^^^^^^^^^^

.. note:: ID: ``cards.race_middle``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "middle",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.boolq:

----------

boolq
^^^^^

.. note:: ID: ``cards.boolq``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "boolq",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "answers": [
                          "yes",
                          "false"
                      ],
                      "topic": "boolean questions"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": {
                      "answer": "str"
                  },
                  "type": "cast_fields"
              },
              {
                  "field_to_field": {
                      "answer": "label",
                      "answers": "answers",
                      "passage": "context",
                      "question": "question",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "answers",
                  "separator": ",",
                  "to_field": "answers",
                  "type": "join_str"
              }
          ],
          "task": {
              "inputs": [
                  "question",
                  "label",
                  "context",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "clean": {
                  "input_format": "Context: {context}\nQuestion: {question}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.piqa_high:

----------

piqa_high
^^^^^^^^^

.. note:: ID: ``cards.piqa_high``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n                            {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mrpc:

----------

mrpc
^^^^

.. note:: ID: ``cards.mrpc``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "mrpc",
              "path": "glue",
              "streaming": false,
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.default",
              {
                  "mappers": {
                      "label": {
                          "0": "not equivalent",
                          "1": "equivalent"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "not equivalent",
                          "equivalent"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "sentence2"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.default <splitters.default>`

|
|



.. _.cards.winogrande:

----------

winogrande
^^^^^^^^^^



.. _cards.winogrande.l:

----------

l
"

.. note:: ID: ``cards.winogrande.l``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_l",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.winogrande.m:

----------

m
"

.. note:: ID: ``cards.winogrande.m``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_m",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.winogrande.s:

----------

s
"

.. note:: ID: ``cards.winogrande.s``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_s",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.winogrande.xs:

----------

xs
""

.. note:: ID: ``cards.winogrande.xs``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_xs",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.winogrande.xl:

----------

xl
""

.. note:: ID: ``cards.winogrande.xl``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_xl",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.winogrande.debiased:

----------

debiased
""""""""

.. note:: ID: ``cards.winogrande.debiased``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "winogrande_debiased",
              "path": "winogrande",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "common sense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "option1",
                      "option2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "fields": {
                      "answer": "int"
                  },
                  "type": "cast_fields"
              },
              {
                  "add": -1,
                  "field": "answer",
                  "type": "add_constant"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "sentence": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.rte:

----------

rte
^^^

.. note:: ID: ``cards.rte``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "rte",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "sentence1": "premise",
                      "sentence2": "hypothesis"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": "tasks.nli",
          "templates": "templates.classification.nli.all",
          "type": "task_card"
      }

References: :ref:`tasks.nli <tasks.nli>`, :ref:`templates.classification.nli.all <templates.classification.nli.all>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.qnli:

----------

qnli
^^^^

.. note:: ID: ``cards.qnli``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "qnli",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.large_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "question": "premise",
                      "sentence": "hypothesis"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": "tasks.nli",
          "templates": "templates.classification.nli.all",
          "type": "task_card"
      }

References: :ref:`splitters.large_no_test <splitters.large_no_test>`, :ref:`tasks.nli <tasks.nli>`, :ref:`templates.classification.nli.all <templates.classification.nli.all>`

|
|



.. _cards.wmt_en_fr:

----------

wmt_en_fr
^^^^^^^^^

.. note:: ID: ``cards.wmt_en_fr``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "fr-en",
              "path": "wmt14",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mix": {
                      "test": "test",
                      "train": "train",
                      "validation": "validation"
                  },
                  "type": "split_random_mix"
              },
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "en"
                      ],
                      [
                          "translation/fr",
                          "fr"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              }
          ],
          "task": {
              "inputs": [
                  "en"
              ],
              "metrics": [
                  "metrics.bleu"
              ],
              "outputs": [
                  "fr"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "{en}",
                      "output_format": "{fr}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.bleu <metrics.bleu>`

|
|



.. _cards.mnli:

----------

mnli
^^^^

.. note:: ID: ``cards.mnli``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "mnli",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "validation_matched": "validation"
                  },
                  "type": "rename_splits"
              },
              "splitters.small_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "neutral",
                          "2": "contradiction"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "neutral",
                          "contradiction"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.nli",
          "templates": "templates.classification.nli.all",
          "type": "task_card"
      }

References: :ref:`tasks.nli <tasks.nli>`, :ref:`templates.classification.nli.all <templates.classification.nli.all>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.hellaswag:

----------

hellaswag
^^^^^^^^^

.. note:: ID: ``cards.hellaswag``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "hellaswag",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.large_no_test",
              {
                  "fields": {
                      "numbering": [
                          "0",
                          "1",
                          "2",
                          "3",
                          "4",
                          "5",
                          "6",
                          "7",
                          "8",
                          "9",
                          "10",
                          "11",
                          "12",
                          "13",
                          "14",
                          "15",
                          "16",
                          "17",
                          "18",
                          "19",
                          "20",
                          "21",
                          "22",
                          "23",
                          "24",
                          "25",
                          "26",
                          "27",
                          "28",
                          "29",
                          "30",
                          "31",
                          "32",
                          "33",
                          "34",
                          "35",
                          "36",
                          "37",
                          "38",
                          "39",
                          "40",
                          "41",
                          "42",
                          "43",
                          "44",
                          "45",
                          "46",
                          "47",
                          "48",
                          "49",
                          "50",
                          "51",
                          "52",
                          "53",
                          "54",
                          "55",
                          "56",
                          "57",
                          "58",
                          "59",
                          "60",
                          "61",
                          "62",
                          "63",
                          "64",
                          "65",
                          "66",
                          "67",
                          "68",
                          "69",
                          "70",
                          "71",
                          "72",
                          "73",
                          "74",
                          "75",
                          "76",
                          "77",
                          "78",
                          "79",
                          "80",
                          "81",
                          "82",
                          "83",
                          "84",
                          "85",
                          "86",
                          "87",
                          "88",
                          "89",
                          "90",
                          "91",
                          "92",
                          "93",
                          "94",
                          "95",
                          "96",
                          "97",
                          "98",
                          "99",
                          "100",
                          "101",
                          "102",
                          "103",
                          "104",
                          "105",
                          "106",
                          "107",
                          "108",
                          "109",
                          "110",
                          "111",
                          "112",
                          "113",
                          "114",
                          "115",
                          "116",
                          "117",
                          "118",
                          "119",
                          "120",
                          "121",
                          "122",
                          "123",
                          "124",
                          "125",
                          "126",
                          "127",
                          "128",
                          "129",
                          "130",
                          "131",
                          "132",
                          "133",
                          "134",
                          "135",
                          "136",
                          "137",
                          "138",
                          "139",
                          "140",
                          "141",
                          "142",
                          "143",
                          "144",
                          "145",
                          "146",
                          "147",
                          "148",
                          "149",
                          "150",
                          "151",
                          "152",
                          "153",
                          "154",
                          "155",
                          "156",
                          "157",
                          "158",
                          "159",
                          "160",
                          "161",
                          "162",
                          "163",
                          "164",
                          "165",
                          "166",
                          "167",
                          "168",
                          "169",
                          "170",
                          "171",
                          "172",
                          "173",
                          "174",
                          "175",
                          "176",
                          "177",
                          "178",
                          "179",
                          "180",
                          "181",
                          "182",
                          "183",
                          "184",
                          "185",
                          "186",
                          "187",
                          "188",
                          "189",
                          "190",
                          "191",
                          "192",
                          "193",
                          "194",
                          "195",
                          "196",
                          "197",
                          "198",
                          "199"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "label",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "activity_label": "topic",
                      "ctx": "sentence1",
                      "endings": "_endings",
                      "index": "_index",
                      "numbering": "_numbering"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_endings",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_endings"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.large_no_test <splitters.large_no_test>`

|
|



.. _cards.piqa_all:

----------

piqa_all
^^^^^^^^

.. note:: ID: ``cards.piqa_all``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "all",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n                            {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.race_high:

----------

race_high
^^^^^^^^^

.. note:: ID: ``cards.race_high``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.wmt_en_ro:

----------

wmt_en_ro
^^^^^^^^^

.. note:: ID: ``cards.wmt_en_ro``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "ro-en",
              "path": "wmt16",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mix": {
                      "test": "test",
                      "train": "train",
                      "validation": "validation"
                  },
                  "type": "split_random_mix"
              },
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "en"
                      ],
                      [
                          "translation/ro",
                          "ro"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              }
          ],
          "task": {
              "inputs": [
                  "en"
              ],
              "metrics": [
                  "metrics.bleu"
              ],
              "outputs": [
                  "ro"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "{en}",
                      "output_format": "{ro}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.bleu <metrics.bleu>`

|
|



.. _cards.piqa:

----------

piqa
^^^^

.. note:: ID: ``cards.piqa``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "piqa",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "physical commonsense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "sol1",
                      "sol2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "field_to_field": {
                      "choices": "_choices",
                      "goal": "sentence1",
                      "label": "_label",
                      "numbering": "_numbering",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_label",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_label",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_label",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.stsb:

----------

stsb
^^^^

.. note:: ID: ``cards.stsb``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "stsb",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mix": {
                      "test": "validation",
                      "train": "train[95%]",
                      "validation": "train[5%]"
                  },
                  "type": "split_random_mix"
              }
          ],
          "task": {
              "inputs": [
                  "sentence1",
                  "sentence2"
              ],
              "metrics": [
                  "metrics.spearman"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "Given this sentence: '{sentence1}', on a scale of 1 to 5, how similar in meaning is it to this sentence: '{sentence2}'?",
                      "output_format": "{label}",
                      "quantum": 0.2,
                      "type": "output_quantizing_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.spearman <metrics.spearman>`

|
|



.. _cards.cola:

----------

cola
^^^^

.. note:: ID: ``cards.cola``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "cola",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "unacceptable",
                          "1": "acceptable"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "field_to_field": {
                      "sentence": "text"
                  },
                  "type": "rename_fields"
              },
              {
                  "fields": {
                      "choices": [
                          "unacceptable",
                          "acceptable"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "text"
              ],
              "metrics": [
                  "metrics.matthews_correlation"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": "templates.classification.choices.all",
          "type": "task_card"
      }

References: :ref:`templates.classification.choices.all <templates.classification.choices.all>`, :ref:`metrics.matthews_correlation <metrics.matthews_correlation>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.qqp:

----------

qqp
^^^

.. note:: ID: ``cards.qqp``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "qqp",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.large_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "not duplicated",
                          "1": "duplicated"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "not duplicated",
                          "duplicated"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "question1",
                  "question2"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "Given this question: {question1}, classify if this question: {question2} is {choices}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.large_no_test <splitters.large_no_test>`

|
|



.. _cards.ethos_binary:

----------

ethos_binary
^^^^^^^^^^^^

.. note:: ID: ``cards.ethos_binary``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "instructions": null,
          "loader": {
              "name": "binary",
              "path": "ethos",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "train": "test"
                  },
                  "type": "rename_splits"
              },
              "splitters.test_only",
              {
                  "mappers": {
                      "label": {
                          "0": "not hate speech",
                          "1": "hate speech"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "hate speech",
                          "not hate speech"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "text": "sentence1"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "Given this sentence: {sentence1}. Classify if it contains hatespeech. Choices: {choices}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  },
                  {
                      "input_format": "Does the following sentence contains hatespeech? Answer by choosing one of the options {choices}. sentence: {sentence1}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.test_only <splitters.test_only>`

|
|



.. _.cards.ai2_arc:

----------

ai2_arc
^^^^^^^



.. _cards.ai2_arc.ARC_Easy:

----------

ARC_Easy
""""""""

.. note:: ID: ``cards.ai2_arc.ARC_Easy``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "ARC-Easy",
              "path": "ai2_arc",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "topic": "science"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answerKey": "label",
                      "choices": "choices_struct"
                  },
                  "type": "rename_fields"
              },
              {
                  "field_to_field": {
                      "choices_struct/label": "numbering",
                      "choices_struct/text": "choices"
                  },
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "index_of": "label",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "choices": "_choices",
                      "index": "_index",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.ai2_arc.ARC_Challenge:

----------

ARC_Challenge
"""""""""""""

.. note:: ID: ``cards.ai2_arc.ARC_Challenge``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "ARC-Challenge",
              "path": "ai2_arc",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "topic": "science"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answerKey": "label",
                      "choices": "choices_struct"
                  },
                  "type": "rename_fields"
              },
              {
                  "field_to_field": {
                      "choices_struct/label": "numbering",
                      "choices_struct/text": "choices"
                  },
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "index_of": "label",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "choices": "_choices",
                      "index": "_index",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.sst2:

----------

sst2
^^^^

.. note:: ID: ``cards.sst2``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "sst2",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "negative",
                          "1": "positive"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "negative",
                          "positive"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.one_sent_classification",
          "templates": "templates.one_sent_classification",
          "type": "task_card"
      }

References: :ref:`templates.one_sent_classification <templates.one_sent_classification>`, :ref:`tasks.one_sent_classification <tasks.one_sent_classification>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.openbookQA:

----------

openbookQA
^^^^^^^^^^

.. note:: ID: ``cards.openbookQA``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "vietgpt/openbookqa_en",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "general continuation"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "choices/label": "numbering",
                      "choices/text": "text"
                  },
                  "type": "rename_fields",
                  "use_query": true
              },
              {
                  "index_of": "answerKey",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "index": "_index",
                      "numbering": "_numbering",
                      "question_stem": "sentence1",
                      "text": "_text",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_text",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_text"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.wmt_en_de:

----------

wmt_en_de
^^^^^^^^^

.. note:: ID: ``cards.wmt_en_de``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "de-en",
              "path": "wmt16",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mix": {
                      "test": "test",
                      "train": "train",
                      "validation": "validation"
                  },
                  "type": "split_random_mix"
              },
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "en"
                      ],
                      [
                          "translation/de",
                          "de"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              }
          ],
          "task": {
              "inputs": [
                  "en"
              ],
              "metrics": [
                  "metrics.bleu"
              ],
              "outputs": [
                  "de"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "{en}",
                      "output_format": "{de}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.bleu <metrics.bleu>`

|
|



.. _cards.piqa_middle:

----------

piqa_middle
^^^^^^^^^^^

.. note:: ID: ``cards.piqa_middle``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "middle",
              "path": "race",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "reading comprehension"
                  },
                  "type": "add_fields"
              },
              {
                  "index_of": "answer",
                  "search_in": "numbering",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "article": "context",
                      "index": "_index",
                      "numbering": "_numbering",
                      "options": "_options",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_options",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_options"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n\n                            Context: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\n                            {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.copa:

----------

copa
^^^^

.. note:: ID: ``cards.copa``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "copa",
              "path": "super_glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "commonsense causal reasoning"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "choice1",
                      "choice2"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "field_to_field": {
                      "choices": "_choices",
                      "label": "_label",
                      "numbering": "_numbering",
                      "premise": "context",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_label",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_label",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_label",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "What was the {sentence1} of the following:\n{context}\nAnswers: {choices}\nAnswer:",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _.cards.mmlu:

----------

mmlu
^^^^



.. _cards.mmlu.high_school_biology:

----------

high_school_biology
"""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_biology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_biology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school biology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.business_ethics:

----------

business_ethics
"""""""""""""""

.. note:: ID: ``cards.mmlu.business_ethics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "business_ethics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "business ethics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.world_religions:

----------

world_religions
"""""""""""""""

.. note:: ID: ``cards.mmlu.world_religions``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "world_religions",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "world religions"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_mathematics:

----------

college_mathematics
"""""""""""""""""""

.. note:: ID: ``cards.mmlu.college_mathematics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_mathematics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college mathematics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.electrical_engineering:

----------

electrical_engineering
""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.electrical_engineering``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "electrical_engineering",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "electrical engineering"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_biology:

----------

college_biology
"""""""""""""""

.. note:: ID: ``cards.mmlu.college_biology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_biology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college biology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.computer_security:

----------

computer_security
"""""""""""""""""

.. note:: ID: ``cards.mmlu.computer_security``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "computer_security",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "computer security"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_mathematics:

----------

high_school_mathematics
"""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_mathematics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_mathematics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school mathematics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.marketing:

----------

marketing
"""""""""

.. note:: ID: ``cards.mmlu.marketing``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "marketing",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "marketing"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_medicine:

----------

college_medicine
""""""""""""""""

.. note:: ID: ``cards.mmlu.college_medicine``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_medicine",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college medicine"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.nutrition:

----------

nutrition
"""""""""

.. note:: ID: ``cards.mmlu.nutrition``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "nutrition",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "nutrition"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.virology:

----------

virology
""""""""

.. note:: ID: ``cards.mmlu.virology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "virology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "virology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_chemistry:

----------

high_school_chemistry
"""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_chemistry``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_chemistry",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school chemistry"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.philosophy:

----------

philosophy
""""""""""

.. note:: ID: ``cards.mmlu.philosophy``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "philosophy",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "philosophy"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.prehistory:

----------

prehistory
""""""""""

.. note:: ID: ``cards.mmlu.prehistory``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "prehistory",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "prehistory"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_macroeconomics:

----------

high_school_macroeconomics
""""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_macroeconomics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_macroeconomics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school macroeconomics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.security_studies:

----------

security_studies
""""""""""""""""

.. note:: ID: ``cards.mmlu.security_studies``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "security_studies",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "security studies"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.formal_logic:

----------

formal_logic
""""""""""""

.. note:: ID: ``cards.mmlu.formal_logic``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "formal_logic",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "formal logic"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.anatomy:

----------

anatomy
"""""""

.. note:: ID: ``cards.mmlu.anatomy``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "anatomy",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "anatomy"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_psychology:

----------

high_school_psychology
""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_psychology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_psychology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school psychology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_world_history:

----------

high_school_world_history
"""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_world_history``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_world_history",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school world history"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.astronomy:

----------

astronomy
"""""""""

.. note:: ID: ``cards.mmlu.astronomy``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "astronomy",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "astronomy"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.miscellaneous:

----------

miscellaneous
"""""""""""""

.. note:: ID: ``cards.mmlu.miscellaneous``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "miscellaneous",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "miscellaneous"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.econometrics:

----------

econometrics
""""""""""""

.. note:: ID: ``cards.mmlu.econometrics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "econometrics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "econometrics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_european_history:

----------

high_school_european_history
""""""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_european_history``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_european_history",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school european history"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.machine_learning:

----------

machine_learning
""""""""""""""""

.. note:: ID: ``cards.mmlu.machine_learning``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "machine_learning",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "machine learning"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.global_facts:

----------

global_facts
""""""""""""

.. note:: ID: ``cards.mmlu.global_facts``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "global_facts",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "global facts"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.jurisprudence:

----------

jurisprudence
"""""""""""""

.. note:: ID: ``cards.mmlu.jurisprudence``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "jurisprudence",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "jurisprudence"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.public_relations:

----------

public_relations
""""""""""""""""

.. note:: ID: ``cards.mmlu.public_relations``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "public_relations",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "public relations"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.clinical_knowledge:

----------

clinical_knowledge
""""""""""""""""""

.. note:: ID: ``cards.mmlu.clinical_knowledge``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "clinical_knowledge",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "clinical knowledge"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_computer_science:

----------

college_computer_science
""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.college_computer_science``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_computer_science",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college computer science"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_geography:

----------

high_school_geography
"""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_geography``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_geography",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school geography"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.management:

----------

management
""""""""""

.. note:: ID: ``cards.mmlu.management``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "management",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "management"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.international_law:

----------

international_law
"""""""""""""""""

.. note:: ID: ``cards.mmlu.international_law``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "international_law",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "international law"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.moral_disputes:

----------

moral_disputes
""""""""""""""

.. note:: ID: ``cards.mmlu.moral_disputes``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "moral_disputes",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "moral disputes"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.us_foreign_policy:

----------

us_foreign_policy
"""""""""""""""""

.. note:: ID: ``cards.mmlu.us_foreign_policy``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "us_foreign_policy",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "us foreign policy"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.professional_accounting:

----------

professional_accounting
"""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.professional_accounting``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "professional_accounting",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "professional accounting"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_computer_science:

----------

high_school_computer_science
""""""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_computer_science``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_computer_science",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school computer science"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.moral_scenarios:

----------

moral_scenarios
"""""""""""""""

.. note:: ID: ``cards.mmlu.moral_scenarios``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "moral_scenarios",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "moral scenarios"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.abstract_algebra:

----------

abstract_algebra
""""""""""""""""

.. note:: ID: ``cards.mmlu.abstract_algebra``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "abstract_algebra",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "abstract algebra"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.sociology:

----------

sociology
"""""""""

.. note:: ID: ``cards.mmlu.sociology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "sociology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "sociology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.human_sexuality:

----------

human_sexuality
"""""""""""""""

.. note:: ID: ``cards.mmlu.human_sexuality``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "human_sexuality",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "human sexuality"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_statistics:

----------

high_school_statistics
""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_statistics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_statistics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school statistics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_us_history:

----------

high_school_us_history
""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_us_history``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_us_history",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school us history"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.professional_psychology:

----------

professional_psychology
"""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.professional_psychology``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "professional_psychology",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "professional psychology"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_physics:

----------

high_school_physics
"""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_physics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_physics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school physics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.conceptual_physics:

----------

conceptual_physics
""""""""""""""""""

.. note:: ID: ``cards.mmlu.conceptual_physics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "conceptual_physics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "conceptual physics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_chemistry:

----------

college_chemistry
"""""""""""""""""

.. note:: ID: ``cards.mmlu.college_chemistry``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_chemistry",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college chemistry"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.college_physics:

----------

college_physics
"""""""""""""""

.. note:: ID: ``cards.mmlu.college_physics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "college_physics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "college physics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.elementary_mathematics:

----------

elementary_mathematics
""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.elementary_mathematics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "elementary_mathematics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "elementary mathematics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.logical_fallacies:

----------

logical_fallacies
"""""""""""""""""

.. note:: ID: ``cards.mmlu.logical_fallacies``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "logical_fallacies",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "logical fallacies"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.human_aging:

----------

human_aging
"""""""""""

.. note:: ID: ``cards.mmlu.human_aging``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "human_aging",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "human aging"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.professional_law:

----------

professional_law
""""""""""""""""

.. note:: ID: ``cards.mmlu.professional_law``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "professional_law",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "professional law"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_microeconomics:

----------

high_school_microeconomics
""""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_microeconomics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_microeconomics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school microeconomics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.high_school_government_and_politics:

----------

high_school_government_and_politics
"""""""""""""""""""""""""""""""""""

.. note:: ID: ``cards.mmlu.high_school_government_and_politics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "high_school_government_and_politics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "high school government and politics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.professional_medicine:

----------

professional_medicine
"""""""""""""""""""""

.. note:: ID: ``cards.mmlu.professional_medicine``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "professional_medicine",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "professional medicine"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.mmlu.medical_genetics:

----------

medical_genetics
""""""""""""""""

.. note:: ID: ``cards.mmlu.medical_genetics``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "medical_genetics",
              "path": "cais/mmlu",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "mapper": {
                      "auxiliary_train": "train"
                  },
                  "type": "rename_splits"
              },
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "medical genetics"
                  },
                  "type": "add_fields"
              },
              {
                  "field_to_field": {
                      "answer": "_answer",
                      "choices": "_choices",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_answer",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_answer",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_answer",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nQuestion: {sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _cards.wnli:

----------

wnli
^^^^

.. note:: ID: ``cards.wnli``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "wnli",
              "path": "glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "field_to_field": {
                      "sentence1": "premise",
                      "sentence2": "hypothesis"
                  },
                  "type": "rename_fields"
              },
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.nli",
          "templates": "templates.classification.nli.all",
          "type": "task_card"
      }

References: :ref:`tasks.nli <tasks.nli>`, :ref:`templates.classification.nli.all <templates.classification.nli.all>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _.cards.wmt:

----------

wmt
^^^



.. _cards.wmt.en_ro:

----------

en_ro
"""""

.. note:: ID: ``cards.wmt.en_ro``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "ro-en",
              "path": "wmt16",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "text"
                      ],
                      [
                          "translation/ro",
                          "translation"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "fields": {
                      "source_language": "english",
                      "target_language": "romanian"
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.translation.directed",
          "templates": "templates.translation.directed.all",
          "type": "task_card"
      }

References: :ref:`tasks.translation.directed <tasks.translation.directed>`, :ref:`templates.translation.directed.all <templates.translation.directed.all>`

|
|



.. _cards.wmt.en_fr:

----------

en_fr
"""""

.. note:: ID: ``cards.wmt.en_fr``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "fr-en",
              "path": "wmt14",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "text"
                      ],
                      [
                          "translation/fr",
                          "translation"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "fields": {
                      "source_language": "english",
                      "target_language": "french"
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.translation.directed",
          "templates": "templates.translation.directed.all",
          "type": "task_card"
      }

References: :ref:`tasks.translation.directed <tasks.translation.directed>`, :ref:`templates.translation.directed.all <templates.translation.directed.all>`

|
|



.. _cards.wmt.en_de:

----------

en_de
"""""

.. note:: ID: ``cards.wmt.en_de``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "de-en",
              "path": "wmt16",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "field_to_field": [
                      [
                          "translation/en",
                          "text"
                      ],
                      [
                          "translation/de",
                          "translation"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              },
              {
                  "fields": {
                      "source_language": "english",
                      "target_language": "deutch"
                  },
                  "type": "add_fields"
              }
          ],
          "task": "tasks.translation.directed",
          "templates": "templates.translation.directed.all",
          "type": "task_card"
      }

References: :ref:`tasks.translation.directed <tasks.translation.directed>`, :ref:`templates.translation.directed.all <templates.translation.directed.all>`

|
|



.. _cards.cnn_dailymail:

----------

cnn_dailymail
^^^^^^^^^^^^^

.. note:: ID: ``cards.cnn_dailymail``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "3.0.0",
              "path": "cnn_dailymail",
              "type": "load_hf"
          },
          "preprocess_steps": [],
          "task": {
              "inputs": [
                  "article"
              ],
              "metrics": [
                  "metrics.rouge"
              ],
              "outputs": [
                  "highlights"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "{article}",
                      "output_format": "{highlights}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.rouge <metrics.rouge>`

|
|



.. _cards.squad:

----------

squad
^^^^^

.. note:: ID: ``cards.squad``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "squad",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "field_to_field": [
                      [
                          "answers/text",
                          "answer"
                      ]
                  ],
                  "type": "copy_fields",
                  "use_query": true
              }
          ],
          "task": {
              "inputs": [
                  "context",
                  "question"
              ],
              "metrics": [
                  "metrics.squad"
              ],
              "outputs": [
                  "answer"
              ],
              "type": "form_task"
          },
          "templates": "templates.qa.contextual.all",
          "type": "task_card"
      }

References: :ref:`templates.qa.contextual.all <templates.qa.contextual.all>`, :ref:`metrics.squad <metrics.squad>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.wsc:

----------

wsc
^^^

.. note:: ID: ``cards.wsc``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "name": "wsc",
              "path": "super_glue",
              "type": "load_hf"
          },
          "preprocess_steps": [
              "splitters.small_no_test",
              {
                  "mappers": {
                      "label": {
                          "0": "False",
                          "1": "True"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "False",
                          "True"
                      ]
                  },
                  "type": "add_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "text",
                  "span1_text",
                  "span2_text"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "items": [
                  {
                      "input_format": "Given this sentence: {text} classify if \"{span2_text}\" refers to \"{span1_text}\".",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  }
              ],
              "type": "templates_list"
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`, :ref:`splitters.small_no_test <splitters.small_no_test>`

|
|



.. _cards.sciq:

----------

sciq
^^^^

.. note:: ID: ``cards.sciq``  |  Type: :class:`TaskCard <unitxt.card.TaskCard>`

   .. code-block:: json

      {
          "loader": {
              "path": "sciq",
              "type": "load_hf"
          },
          "preprocess_steps": [
              {
                  "fields": {
                      "numbering": [
                          "A",
                          "B",
                          "C",
                          "D",
                          "E",
                          "F",
                          "G",
                          "H",
                          "I",
                          "J",
                          "K",
                          "L",
                          "M",
                          "N",
                          "O",
                          "P",
                          "Q",
                          "R",
                          "S",
                          "T",
                          "U",
                          "V",
                          "W",
                          "X",
                          "Y",
                          "Z"
                      ],
                      "topic": "physical commonsense"
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "distractor1",
                      "distractor2",
                      "distractor3",
                      "correct_answer"
                  ],
                  "to_field": "choices",
                  "type": "list_field_values"
              },
              {
                  "field": "choices",
                  "type": "shuffle_field_values"
              },
              {
                  "index_of": "correct_answer",
                  "search_in": "choices",
                  "to_field": "index",
                  "type": "index_of"
              },
              {
                  "field_to_field": {
                      "choices": "_choices",
                      "index": "_index",
                      "numbering": "_numbering",
                      "question": "sentence1",
                      "support": "context",
                      "topic": "topic"
                  },
                  "type": "rename_fields"
              },
              {
                  "field": "_numbering",
                  "index": "_index",
                  "to_field": "number",
                  "type": "take_by_field"
              },
              {
                  "field": "_choices",
                  "index": "_index",
                  "to_field": "answer",
                  "type": "take_by_field"
              },
              {
                  "fields": [
                      "_numbering",
                      "_choices"
                  ],
                  "to_field": "choices",
                  "type": "zip_field_values"
              },
              {
                  "field": "choices/*",
                  "process_every_value": true,
                  "separator": ". ",
                  "to_field": "choices_list",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "index": "_index",
                  "to_field": "number_and_answer",
                  "type": "take_by_field"
              },
              {
                  "field": "choices/*/0",
                  "separator": ",",
                  "to_field": "numbers",
                  "type": "join_str",
                  "use_query": true
              },
              {
                  "field": "choices_list",
                  "separator": " ",
                  "to_field": "choices",
                  "type": "join_str"
              },
              {
                  "field_to_field": {
                      "number": "label"
                  },
                  "type": "rename_fields"
              }
          ],
          "task": {
              "inputs": [
                  "choices",
                  "sentence1",
                  "numbers",
                  "topic",
                  "context"
              ],
              "metrics": [
                  "metrics.accuracy"
              ],
              "outputs": [
                  "label"
              ],
              "type": "form_task"
          },
          "templates": {
              "fm-eval": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}\nChoose from {numbers}\nAnswers: {choices}\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "helm": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n\nContext: {context}\nQuestion: {context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "lm_eval_harness": {
                  "input_format": "Context: {context}\nQuestion: {context}\n{sentence1}.\nChoices:\n{choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              },
              "original": {
                  "input_format": "The following are multiple choice questions (with answers) about {topic}.\n{context}\n{sentence1}.\nAnswers: {choices}.\nAnswer:",
                  "output_format": "{label}",
                  "type": "input_output_template"
              }
          },
          "type": "task_card"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _.instructions:

----------

instructions
------------



.. _.instructions.models:

----------

models
^^^^^^



.. _instructions.models.llama:

----------

llama
"""""

.. note:: ID: ``instructions.models.llama``  |  Type: :class:`TextualInstruction <unitxt.instructions.TextualInstruction>`

   .. code-block:: json

      {
          "text": "You are a very smart model. solve the following.",
          "type": "textual_instruction"
      }


|
|



.. _.tasks:

----------

tasks
-----



.. _.tasks.translation:

----------

translation
^^^^^^^^^^^



.. _tasks.translation.directed:

----------

directed
""""""""

.. note:: ID: ``tasks.translation.directed``  |  Type: :class:`FormTask <unitxt.task.FormTask>`

   .. code-block:: json

      {
          "inputs": [
              "text",
              "source_language",
              "target_language"
          ],
          "metrics": [
              "metrics.bleu"
          ],
          "outputs": [
              "translation"
          ],
          "type": "form_task"
      }

References: :ref:`metrics.bleu <metrics.bleu>`

|
|



.. _tasks.nli:

----------

nli
^^^

.. note:: ID: ``tasks.nli``  |  Type: :class:`FormTask <unitxt.task.FormTask>`

   .. code-block:: json

      {
          "inputs": [
              "choices",
              "premise",
              "hypothesis"
          ],
          "metrics": [
              "metrics.accuracy"
          ],
          "outputs": [
              "label"
          ],
          "type": "form_task"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _tasks.one_sent_classification:

----------

one_sent_classification
^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``tasks.one_sent_classification``  |  Type: :class:`FormTask <unitxt.task.FormTask>`

   .. code-block:: json

      {
          "inputs": [
              "choices",
              "sentence"
          ],
          "metrics": [
              "metrics.accuracy"
          ],
          "outputs": [
              "label"
          ],
          "type": "form_task"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _.recipes:

----------

recipes
-------



.. _recipes.wnli_5_shot:

----------

wnli_5_shot
^^^^^^^^^^^

.. note:: ID: ``recipes.wnli_5_shot``  |  Type: :class:`SequentialRecipe <unitxt.recipe.SequentialRecipe>`

   .. code-block:: json

      {
          "steps": [
              {
                  "name": "wnli",
                  "path": "glue",
                  "type": "load_hf"
              },
              {
                  "mix": {
                      "test": "validation",
                      "train": "train[95%]",
                      "validation": "train[5%]"
                  },
                  "type": "split_random_mix"
              },
              {
                  "slices": {
                      "demos_pool": "train[:100]",
                      "test": "test",
                      "train": "train[100:]",
                      "validation": "validation"
                  },
                  "type": "slice_split"
              },
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "fields": [
                      "choices"
                  ],
                  "type": "normalize_list_fields"
              },
              {
                  "inputs": [
                      "choices",
                      "sentence1",
                      "sentence2"
                  ],
                  "metrics": [
                      "metrics.accuracy"
                  ],
                  "outputs": [
                      "label"
                  ],
                  "type": "form_task"
              },
              {
                  "sampler": {
                      "sample_size": 5,
                      "type": "random_sampler"
                  },
                  "source_stream": "demos_pool",
                  "target_field": "demos",
                  "type": "spread_split"
              },
              {
                  "demos_field": "demos",
                  "instruction": {
                      "text": "classify if this sentence is entailment or not entailment.",
                      "type": "textual_instruction"
                  },
                  "template": {
                      "input_format": "Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  },
                  "type": "render_templated_icl"
              }
          ],
          "type": "sequential_recipe"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _recipes.wnli_3_shot:

----------

wnli_3_shot
^^^^^^^^^^^

.. note:: ID: ``recipes.wnli_3_shot``  |  Type: :class:`CommonRecipe <unitxt.common.CommonRecipe>`

   .. code-block:: json

      {
          "card": "cards.wnli",
          "demos_pool_size": 100,
          "num_demos": 3,
          "template_item": 0,
          "type": "common_recipe"
      }

References: :ref:`cards.wnli <cards.wnli>`

|
|



.. _recipes.wnli_fixed:

----------

wnli_fixed
^^^^^^^^^^

.. note:: ID: ``recipes.wnli_fixed``  |  Type: :class:`SequentialRecipe <unitxt.recipe.SequentialRecipe>`

   .. code-block:: json

      {
          "steps": [
              {
                  "name": "wnli",
                  "path": "glue",
                  "type": "load_hf"
              },
              {
                  "mix": {
                      "test": "validation",
                      "train": "train[95%]",
                      "validation": "train[5%]"
                  },
                  "type": "split_random_mix"
              },
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ]
                  },
                  "type": "add_fields"
              },
              {
                  "inputs": [
                      "choices",
                      "sentence1",
                      "sentence2"
                  ],
                  "metrics": [
                      "metrics.accuracy"
                  ],
                  "outputs": [
                      "label"
                  ],
                  "type": "form_task"
              },
              {
                  "template": {
                      "input_format": "Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.",
                      "output_format": "{label}",
                      "type": "input_output_template"
                  },
                  "type": "render_format_template"
              }
          ],
          "type": "sequential_recipe"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _recipes.wnli:

----------

wnli
^^^^

.. note:: ID: ``recipes.wnli``  |  Type: :class:`SequentialRecipe <unitxt.recipe.SequentialRecipe>`

   .. code-block:: json

      {
          "steps": [
              {
                  "name": "wnli",
                  "path": "glue",
                  "type": "load_hf"
              },
              {
                  "mix": {
                      "test": "validation",
                      "train": "train[95%]",
                      "validation": "train[5%]"
                  },
                  "type": "split_random_mix"
              },
              {
                  "mappers": {
                      "label": {
                          "0": "entailment",
                          "1": "not entailment"
                      }
                  },
                  "type": "map_instance_values"
              },
              {
                  "fields": {
                      "choices": [
                          "entailment",
                          "not entailment"
                      ],
                      "instruction": "classify the relationship between the two sentences from the choices."
                  },
                  "type": "add_fields"
              },
              {
                  "inputs": [
                      "choices",
                      "instruction",
                      "sentence1",
                      "sentence2"
                  ],
                  "metrics": [
                      "metrics.accuracy"
                  ],
                  "outputs": [
                      "label"
                  ],
                  "type": "form_task"
              },
              {
                  "type": "render_auto_format_template"
              }
          ],
          "type": "sequential_recipe"
      }

References: :ref:`metrics.accuracy <metrics.accuracy>`

|
|



.. _.benchmarks:

----------

benchmarks
----------



.. _benchmarks.glue:

----------

glue
^^^^

.. note:: ID: ``benchmarks.glue``  |  Type: :class:`WeightedFusion <unitxt.fusion.WeightedFusion>`

   .. code-block:: json

      {
          "include_splits": null,
          "origins": [
              "recipes.wnli_3_shot",
              "recipes.wnli_3_shot"
          ],
          "total_examples": 4,
          "type": "weighted_fusion",
          "weights": [
              1,
              1
          ]
      }

References: :ref:`recipes.wnli_3_shot <recipes.wnli_3_shot>`

|
|



.. _.processors:

----------

processors
----------



.. _processors.dict_of_lists_to_value_key_pairs:

----------

dict_of_lists_to_value_key_pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.dict_of_lists_to_value_key_pairs``  |  Type: :class:`DictOfListsToPairs <unitxt.processors.DictOfListsToPairs>`

   .. code-block:: json

      {
          "position_key_before_value": false,
          "type": "dict_of_lists_to_pairs"
      }


|
|



.. _processors.load_json:

----------

load_json
^^^^^^^^^

.. note:: ID: ``processors.load_json``  |  Type: :class:`LoadJson <unitxt.processors.LoadJson>`

   .. code-block:: json

      {
          "type": "load_json"
      }


|
|



.. _processors.to_list_by_comma:

----------

to_list_by_comma
^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.to_list_by_comma``  |  Type: :class:`ToListByComma <unitxt.processors.ToListByComma>`

   .. code-block:: json

      {
          "type": "to_list_by_comma"
      }


|
|



.. _processors.to_span_label_pairs:

----------

to_span_label_pairs
^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.to_span_label_pairs``  |  Type: :class:`RegexParser <unitxt.processors.RegexParser>`

   .. code-block:: json

      {
          "regex": "\\s*((?:[^,:\\\\]|\\\\.)+?)\\s*:\\s*((?:[^,:\\\\]|\\\\.)+?)\\s*(?=,|$)",
          "type": "regex_parser"
      }


|
|



.. _processors.to_pairs:

----------

to_pairs
^^^^^^^^

.. note:: ID: ``processors.to_pairs``  |  Type: :class:`RegexParser <unitxt.processors.RegexParser>`

   .. code-block:: json

      {
          "regex": "(\\w+):(\\w+)",
          "type": "regex_parser"
      }


|
|



.. _processors.to_string_stripped:

----------

to_string_stripped
^^^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.to_string_stripped``  |  Type: :class:`ToStringStripped <unitxt.processors.ToStringStripped>`

   .. code-block:: json

      {
          "type": "to_string_stripped"
      }


|
|



.. _processors.list_to_empty_entity_tuples:

----------

list_to_empty_entity_tuples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.list_to_empty_entity_tuples``  |  Type: :class:`ListToEmptyEntitiesTuples <unitxt.processors.ListToEmptyEntitiesTuples>`

   .. code-block:: json

      {
          "type": "list_to_empty_entities_tuples"
      }


|
|



.. _processors.to_span_label_pairs_surface_only:

----------

to_span_label_pairs_surface_only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``processors.to_span_label_pairs_surface_only``  |  Type: :class:`RegexParser <unitxt.processors.RegexParser>`

   .. code-block:: json

      {
          "regex": "\\s*((?:\\\\.|[^,])+?)\\s*(?:,|$)()",
          "termination_regex": "^\\s*None\\s*$",
          "type": "regex_parser"
      }


|
|



.. _processors.to_string:

----------

to_string
^^^^^^^^^

.. note:: ID: ``processors.to_string``  |  Type: :class:`ToString <unitxt.processors.ToString>`

   .. code-block:: json

      {
          "type": "to_string"
      }


|
|



.. _.splitters:

----------

splitters
---------



.. _splitters.large_no_test:

----------

large_no_test
^^^^^^^^^^^^^

.. note:: ID: ``splitters.large_no_test``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "validation",
              "train": "train[99%]",
              "validation": "train[1%]"
          },
          "type": "split_random_mix"
      }


|
|



.. _splitters.small_no_dev:

----------

small_no_dev
^^^^^^^^^^^^

.. note:: ID: ``splitters.small_no_dev``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "test",
              "train": "train[95%]",
              "validation": "train[5%]"
          },
          "type": "split_random_mix"
      }


|
|



.. _splitters.default:

----------

default
^^^^^^^

.. note:: ID: ``splitters.default``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "test",
              "train": "train",
              "validation": "validation"
          },
          "type": "split_random_mix"
      }


|
|



.. _splitters.diverse_labels_sampler:

----------

diverse_labels_sampler
^^^^^^^^^^^^^^^^^^^^^^

.. note:: ID: ``splitters.diverse_labels_sampler``  |  Type: :class:`DiverseLabelsSampler <unitxt.splitters.DiverseLabelsSampler>`

   .. code-block:: json

      {
          "type": "diverse_labels_sampler"
      }


|
|



.. _splitters.large_no_dev:

----------

large_no_dev
^^^^^^^^^^^^

.. note:: ID: ``splitters.large_no_dev``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "test",
              "train": "train[99%]",
              "validation": "train[1%]"
          },
          "type": "split_random_mix"
      }


|
|



.. _splitters.test_only:

----------

test_only
^^^^^^^^^

.. note:: ID: ``splitters.test_only``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "test",
              "train": "test[0%]",
              "validation": "test[0%]"
          },
          "type": "split_random_mix"
      }


|
|



.. _splitters.small_no_test:

----------

small_no_test
^^^^^^^^^^^^^

.. note:: ID: ``splitters.small_no_test``  |  Type: :class:`SplitRandomMix <unitxt.splitters.SplitRandomMix>`

   .. code-block:: json

      {
          "mix": {
              "test": "validation",
              "train": "train[95%]",
              "validation": "train[5%]"
          },
          "type": "split_random_mix"
      }


|
|

