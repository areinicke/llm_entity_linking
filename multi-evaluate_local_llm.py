import subprocess
import os

def run_evaluation(cmd):
    """
    Function to run the evaluation command.
    """
    print(f"Running command: {cmd}")
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "1,2"
    subprocess.run(cmd, env=my_env)


def main():
    base_command = ["python3", "./Implementation/evaluate.py",
                    #"--LLM_model_name", "gemini-2.5-flash",
                    #"--LLM_model_name", "gpt-4o-mini-2024-07-18",
                    "--LLM_model_name", "--2--",
                    "--skip_dev",
                    "--rate_limit_timeout", "0",
                    "--local_llm",
                    "--batching",
                    "--max_output_tokens", "3000",
                    "--corpus", "--1--",
                    "--top_k", "10",
                    "--LLM_verbalization_strategy", "categories",
                    "--LLM_selection_strategy", "difference_abs",
                    "--threshold", "1.0",
                    "--iterations", "2",
                    "--candidate_ordering", "sorted",
                    "--spans_per_prompt", "5",
                    "--overview_save_path", "Results/Multi-Evaluate-Overview/finals_local_llm-overview.txt",
                    "--overview_run_name", "--x--"
                   ]

    parameter_change = {#"--1--": ["AIDA","tweeki", "reddit-comments", "shadowlinks-shadow", "shadowlinks-tail"],
                        #"--1--": ["msnbc", "ace2004", "aquaint"],
                        #"--1--": ["AIDA","cweb", "tweeki", "reddit-comments", "shadowlinks-shadow", "shadowlinks-tail", "shadowlinks-top", "reddit-posts", "wned-wiki"],
                        #"--1--": ["tweeki", "reddit-comments", "reddit-posts", "shadowlinks-shadow", "shadowlinks-tail"],
                        "--1--": ["AIDA"],
                        "--2--": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]
                        #"--2--": ["gpt-4o-mini-2024-07-18"]
                        #"--2--": ["gemini-2.5-pro"]
                       }
    
    run_repeat = 3 # Number of times to repeat each configuration
    run_name = "finals_local_llm"

    for param1 in parameter_change["--1--"]:
        for param2 in parameter_change["--2--"]:
            for i in range(run_repeat):
                # Deep copy the base command
                cmd = base_command.copy()
                # Replace parameters
                cmd[cmd.index("--1--")] = str(param1)
                cmd[cmd.index("--2--")] = str(param2)
                cmd[cmd.index("--x--")] = f"{param1}--{run_name}_{param2}--run{i+1}"
                run_evaluation(cmd)


if __name__ == "__main__":
    main()