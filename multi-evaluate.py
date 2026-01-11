import subprocess

def run_evaluation(cmd):
    """
    Function to run the evaluation command.
    """
    print(f"Running command: {cmd}")
    subprocess.run(cmd)


def main():
    base_command = ["python3", "./Implementation/evaluate.py",
                    #"--LLM_model_name", "gemini-2.5-flash",
                    #"--LLM_model_name", "gpt-4o-mini-2024-07-18",
                    "--LLM_model_name", "--2--",
                    "--max_output_tokens", "2000",
                    "--reasoning", "low",
                    "--skip_dev",
                    "--rate_limit_timeout", "20",
                    "--corpus", "--1--",
                    "--top_k", "10",
                    "--LLM_verbalization_strategy", "categories",
                    "--LLM_verbalization_candidate_strategy", "categories",
                    "--LLM_selection_strategy", "difference_abs",
                    "--threshold", "1.0",
                    "--iterations", "2",
                    "--candidate_ordering", "sorted",
                    "--spans_per_prompt", "5",
                    "--overview_save_path", "Results/Multi-Evaluate-Overview/add_datasets_gpt-overview.txt",
                    "--overview_run_name", "--x--"
                   ]

    parameter_change = {#"--1--": ["AIDA","tweeki", "reddit-comments", "shadowlinks-shadow", "shadowlinks-tail"],
                        "--1--": ["msnbc", "ace2004", "aquaint"],
                        #"--1--": ["AIDA", "cweb", "tweeki", "reddit-comments", "shadowlinks-shadow", "shadowlinks-tail", "shadowlinks-top", "reddit-posts", "wned-wiki"],
                        #"--1--": ["tweeki", "reddit-comments", "reddit-posts", "shadowlinks-shadow", "shadowlinks-tail"],
                        #"--1--": ["shadowlinks-shadow", "shadowlinks-tail", "shadowlinks-top", "wned-wiki"],
                        #"--2--": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]
                        "--2--": ["gpt-5-mini-2025-08-07"]
                        #"--2--": ["gemini-2.5-flash-lite"]
                       }
    
    run_repeat = 3 # Number of times to repeat each configuration
    run_name = "add-datasets-gpt"  # Base name for the run
    override_name = ""  # If not empty, use this string instead of param2 in run naming

    for param1 in parameter_change["--1--"]:
        for param2 in parameter_change["--2--"]:
            for i in range(run_repeat):
                # Deep copy the base command
                cmd = base_command.copy()
                # Replace parameters
                cmd[cmd.index("--1--")] = str(param1)
                cmd[cmd.index("--2--")] = str(param2)
                cmd[cmd.index("--x--")] = f"{param1}--{run_name}_{param2 if override_name == '' else override_name}--run{i+1}"
                run_evaluation(cmd)


if __name__ == "__main__":
    main()
