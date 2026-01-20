# Evaluation Repository for the Thesis "Boosting Entity Linking with LLMs: A Confidence-Guided Hybrid Strategy"

This repository contains the code that was used to evaluate the implementation of our Dual Encoder (found at https://github.com/areinicke/flair) on the ZELDA benchmarks.


---
## Performing a benchmark

In this section, we will explain how to perform a benchmark on a dataset included in ZELDA.  
  
### Getting necessary files:  
  
Due to the size of some required files, not all of them are included in this repository. You will need to download two additional files found [here](https://box.hu-berlin.de/d/696f25beca8746009b0b/).
These files need to be placed in the following folders:  
- ```final-model.pt``` must be placed in ```Implementation/Models/final-model/```
- ```verbalizations_structured.json``` must be placed in the root directory of this project.


### Setting up the environment

You will need to install the following libraries for Python. We have used ```Python 3.10.16``` so it is recommended that you use the same.  

- Our modified version of flair. Install using ```pip install git+https://github.com/areinicke/flair.git@dual_encoder_similarity_loss```  
- All packages listed in https://github.com/areinicke/flair/blob/dual_encoder_similarity_loss/requirements.txt
- vllm
- gradio-client
- openai
- google-genai


### Running benchmarks

The evaluate script found at ```Implementation/evaluate.py``` is the primary script used to run benchmarks on our implementation. When running the script, you can and must set a number of parameters.  
  
The parameters are as follows:  
```--model_path <path>``` - Optional - Path to the ML model (should be a .pt file. Default is ```Implementation/Models/final-model/final-model.pt```  
```--corpus <AIDA|tweeki|reddit-posts|reddit-comments|wned-wiki|cweb|shadowlinks-tail|shadowlinks-shadow|shadowlinks-top|ace2004|aquaint|msnbc|ZELDA>``` - Optional - Name of the corpus. Default is ```AIDA```  
```--LLM_model_name <model name>``` - REQUIRED - Name of the LLM to use. Use the exact model name from Huggingface, OpenAI or Google. If using a model from OpenAI or Google, make sure there is a .env file in th root directory containing api keys with values ```GOOGLE_API_KEY``` or ```OPENAI_API_KEY```.  
```--local_llm``` - Include this parameter when running a model locally  
```--iterations <number if iterations>``` - Optional - Number of LLM iterations to perform. Default is ```3```  
```--threshold <value>``` - Optional - Value for the Threshold. Default is ```0.8```  
```--LLM_selection_strategy <similarity|similarity_abs|difference|difference_abs|both>``` - Optional - Strategy to use for selecting spans for prompting. Default is ```similarity```  
```--LLM_verbalization_strategy <description|categories|description+categories|none>``` - Optional - How to verbalize spans. Default is ```description```  
```--top_k <value>``` - Optional - Number of candidates to give to the LLM. Default is ```5```  
```--spans_per_prompt <value>``` - Optional - Maximum number of spans per prompt. Default is ```5```  
```--candidate_ordering <random|sorted>``` - Optional - Determines whether candidates are given sorted or in random order to the LLM. Default is ```sorted```  
```--rate_limit_timeout <value in seconds``` - Optional - Number of seconds to wait between each prompting phase to avoid rate limiting. This is only relevant when using an API LLM. Default is ```10```  
```--skip_dev```- Optional - Set this parameter to skip evaluating on the trains et of datasets should one be present. Recommended  
```--max_output_tokens <value>``` - Optional - Maximum number of output tokens a prompt can have. Default is ```200```. When using a reasoning LLM, this must be increased.  
```--reasoning <none|low|medium|high>``` - Optional - Amount of reasoning to use for reasoning LLMs. Only supported and must be set for OpenAI reasoning LLMs. Default is ```none```  

(Note: This is only a selection of the parameter. To view all parameters, please take a look at ```Implementation/evaluate.py```.)  

To perform a very basic benchmark you can simply run:  
```python3 Implementation/evaluate.py --LLM_model_name gemini-2.5-flash-lite --corpus AIDA --skip_dev```  
  
To perform a benchmark on a dataset with the final parameter set we have decided on in the thesis, you can run:
```python3 Implementation/evaluate.py --LLM_model_name gemini-2.5-flash-lite --corpus AIDA --skip_dev --top_k 10 --threshold 1 --LLM_selection_strategy difference_abs --LLM_verbalization_strategy categories --iterations 2 --spans_per_prompt 5```


### Viewing the results
The results of each run are saved to ```Results/<dataset>/<LLM_model_type>/<LLM_model_name>/<timestamp of run>/```. We already perform some basic statistics on the benchmark results and save them in the same folder.


---

### Purpose of .py and .ipynb files in root directory  
These files are used to extract a variety of data from benchmark results or run benchmarks using multiple values in sequence. We will not explain each file's purpose here as they are not required to perform benchmarks.
