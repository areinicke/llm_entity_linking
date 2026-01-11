
from vllm import LLM, SamplingParams
import os

class run_args():
    def __init__(self):
        self.config = None
        self.corpus = ""
        self.model_path = "./Implementation/Models/final-model/final-model.pt"
        self.LLM_model_name = ["RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"]
        self.local_LLM = True
        self.batching = True
        self.regenerate_sentences = False
        self.iterations = 2
        self.threshold = 1.0
        self.LLM_strategy = "default"
        self.LLM_selection_strategy = "difference_abs"
        self.LLM_verbalization_strategy = "categories"
        self.LLM_verbalization_candidate_strategy = "same"
        self.verbalization_path = "./verbalizations_structured.json"
        self.top_k = 10
        self.dynamic_top_k = False
        self.min_batch = -1
        self.max_batch = -1
        self.disable_LLM = False
        self.rate_limit_timeout = 0
        self.skip_dev = True
        self.skip_test = False
        self.max_output_tokens = 5000
        self.spans_per_prompt = 5
        self.random_candidate_order = False
        self.candidate_ordering = "sorted"
        self.reasoning = "none"
        self.num_agents = 1
        self.allow_extra_iteration = False
        self.overview_save_path = ""
        self.overview_run_name = ""


def main():
    parameters = run_args()

    corpus_list = ["AIDA", "tweeki", "reddit-posts", "reddit-comments", "shadowlinks-shadow", "shadowlinks-tail", "shadowlinks-top", "wned-wiki", "cweb"]
    #corpus_list = ["cweb"]
    num_runs = 1
    run_base_name = "finals_local_llm2"
    
    #local_LLM = LLM(model=parameters.LLM_model_name[0], max_model_len=30000)    # DeepSeek

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    #local_LLM = LLM(model=parameters.LLM_model_name[0], max_model_len=30000, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=2) # Nemotron
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    local_LLM = LLM(model=parameters.LLM_model_name[0], max_model_len=35000, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=2) # Scout 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    from evaluate import sanity_check_and_set_arguments, main as evaluate_main

    for corpus in corpus_list:
        parameters.corpus = corpus

        for i in range(num_runs):
            try:
                parameters.overview_run_name = f"{corpus}--{run_base_name}_{parameters.LLM_model_name[0]}--run{i+1}"
                parameters.overview_save_path = f"Results/Multi-Evaluate-Overview/{run_base_name}-overview.txt"
                sanity_check_and_set_arguments(parameters)
                evaluate_main(parameters, local_LLM=local_LLM)
            except Exception as e:
                print(f"Error during evaluation of corpus {corpus}, run {i+1}: {e}")
                continue






if __name__ == "__main__":
    main()
