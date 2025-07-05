#!/bin/bash

######################################################################################################
# BlueBench is an open-source benchmark developed by domain experts to represent required needs of   #
# Enterprise users. It is constructed using state-of-the-art benchmarking methodologies to ensure    #
# validity, robustness, and efficiency by utilizing unitxt’s abilities for dynamic and flexible text #
# processing. As a dynamic and evolving benchmark, BlueBench currently encompasses diverse domains   #
# such as legal, finance, customer support, and news. It also evaluates a range of capabilities,     #
# including RAG, pro-social behavior, summarization, and chatbot performance, with additional tasks  #
# and domains to be integrated over time.                                                            #
#                                                                                                    #
# Further details: https://ibm.biz/bluebench                                                         #
######################################################################################################

if [ $# -eq 0 ]
  then
    echo "Usage: evaluate_bluebench.sh model-to-evaluate-in-litellm-format"
    exit 1
fi

unitxt-evaluate \
    --tasks "benchmarks.bluebench" \
    --model cross_provider \
    --model_args "model_name=$1,max_tokens=1024" \
    --output_path ./results/bluebench \
    --log_samples \
    --trust_remote_code \
    --batch_size 8 \
    --verbosity ERROR

unitxt-summarize ./results/bluebench
