from flair.datasets import NEL_ENGLISH_AIDA, ZELDA, NEL_ENGLISH_REDDIT, NEL_ENGLISH_TWEEKI, ColumnCorpus
from flair.models import DualEncoderEntityDisambiguation, GreedyDualEncoderEntityDisambiguation, LLMDualEncoderEntityDisambiguation
from pathlib import Path
import flair
import re
import os
import sys
from datetime import datetime
import argparse
from dotenv import load_dotenv
import json

from calculate_statistics import calc_and_write_statistics
from util import PausableTimer

from vllm import LLM, SamplingParams

flair.device = "cuda:0"


def load_api_key(llm_type):
    load_dotenv()  # Load environment variables from .env file
    if llm_type == "Google":
        key = os.environ.get("GOOGLE_API_KEY")
        return key
    elif llm_type == "OpenAI":
        key = os.environ.get("OPENAI_API_KEY")
        return key
    elif llm_type == "HU":
        return None
    elif llm_type == "LOCAL":
        return None
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Supported types are: Google, OpenAI, HU.")

def identify_specifications_from_model_identifier(model_args_path):
    with open(model_args_path) as f:
        model_identifier = f.read().strip()
    
    model_args = {}
    model_args["document_level"] = True if "docTrue" in model_identifier else False
    model_args["greedy"] = False if "greedyFalse" in model_identifier else True

    if "mm" in model_identifier:
        model_args["similarity_metric"] = "mm"
    elif "cosine" in model_identifier:
        model_args["similarity_metric"] = "cosine"
    else:
        model_args["similarity_metric"] = "euclidean"

    if "first_last" in model_identifier:
        model_args["embedding_pooling"] = "first_last"
    elif "first" in model_identifier:
        model_args["embedding_pooling"] = "first_last"
    elif "mean" in model_identifier:
        model_args["embedding_pooling"] = "mean"

    # # this is not important for now, might be needed in case we change label set
    # verbalization_strategy_match = re.search(r"LabelUse(.*?)-maxLen", model_identifier)
    # if verbalization_strategy_match:
    #     verbalization_strategy = verbalization_strategy_match.group(1)
    # else:
    #     verbalization_strategy_match = re.search(r"V(.*?)-maxLen", model_identifier)
    #     verbalization_strategy = verbalization_strategy_match.group(1)
    # model_args["verbalization_strategy"] = verbalization_strategy

    max_len_match = re.search(r"maxLen(\d+)", model_identifier)
    if max_len_match:
        max_len = int(max_len_match.group(1))  # Convert the captured group to an integer
    else:
        max_len = 50
    model_args["max_len"] = max_len

    return model_args

def predict(model, corpus, args, model_args):

    print("Starting Predictions using the following specifications:")
    print("Similarity metric:", model.similarity_metric.metric_to_use)
    print("Loss:", model.loss_function)
    print("Embedding Pooling:", model.embedding_pooling)
    print("Document:", model_args["document_level"])
    #print("Verbalization Strategy:", model_args["verbalization_strategy"])
    print("Pooling:", model.embedding_pooling)
    print("Max Len:", model_args["max_len"])
    print("Label Type:", model.label_type)
    print()
    print(f"LLM Model: {model.LLM_model_type} - {model.LLM_model_name}")
    print("LLM Strategy:", model.LLM_strategy)
    print("LLM Selection Strategy:", model.LLM_selection_strategy)
    print("LLM Verbalization Strategy:", model.LLM_verbalization_strategy)
    print("LLM Verbalization Candidate Strategy:", model.LLM_verbalization_candidate_strategy)
    print("LLM Iterations:", model.iterations)
    print("LLM Top K:", model.top_k)
    print("LLM Dynamic Top K:", model.dynamic_top_k)
    print("LLM Batching:", model.batching)
    print("LLM Regenerate Sentences:", model.regenerate_sentences)
    print("LLM Threshold:", model.threshold)
    print(f"Batch Range: {args.min_batch} to {args.max_batch}")
    print("------------------------------------------------------------------")

    evaluate_on = {}
    if not args.skip_dev:
        evaluate_on["dev"] = corpus.dev
    if not args.skip_test:
        evaluate_on["test"] = corpus.test

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory for evaluation results
    eval_path = Path(f"./Results/{args.corpus}/{model.LLM_model_type[0] if len(model.LLM_model_type) == 1 else 'Multi'}/{model.LLM_model_name[0] if len(model.LLM_model_name) == 1 else 'Multi'}/{timestamp}")
    if not eval_path.exists():
        eval_path.mkdir(parents=True, exist_ok=True)

    # Create a directory for sentence cache if it doesn't exist
    sentence_cache_path = Path(f"./sentence_cache/{args.corpus}")
    if not sentence_cache_path.exists():
        sentence_cache_path.mkdir(parents=True, exist_ok=True)

    # Write Run parameters to file
    with open(eval_path / "run_parameters.json", "w") as f:
        args_dict = vars(args)
        args_dict["Prompt_Version"] = 3     # Current version of the prompt. Increase by 1 if you edit the prompt significantly
        combined_dict = {**args_dict, **model_args}
        json.dump(combined_dict, f, indent=4)

    for name, sentences in evaluate_on.items():
        print(f"Evaluating on {name} set with {len(sentences)} sentences.")

        results[name] = model.evaluate([s for s in sentences],
                                       out_path=eval_path / f"predictions_{name}.txt",
                                       gold_label_type=model._label_type,
                                       return_loss=True,
                                       corpus_name=args.corpus,
                                       min_batch=args.min_batch,  # Set to -1 to evaluate all batches
                                       max_batch=args.max_batch,  # Set to -1 to evaluate all batches
                                       disable_LLM=args.disable_LLM,  # Disable LLM predictions for this evaluation
                                       rate_limit_timeout=args.rate_limit_timeout,  # Time to wait to avoid rate limiting for the LLM
                                       max_output_tokens=args.max_output_tokens,  # Maximum number of output tokens for the LLM
                                       spans_per_prompt=args.spans_per_prompt,  # Number of spans per prompt for LLM predictions
                                       random_candidate_order=args.random_candidate_order,  # Randomize the order of candidates for LLM predictions
                                       allow_extra_iteration=args.allow_extra_iteration,  # Allow an extra iteration for the LLM predictions for spans which could were not successfully predicted in the final iteration
                                      )

        # Save Results
        with open(eval_path / f"results_{name}.txt", "w") as f:
            f.write(str(results[name]))

        overview_string = calc_and_write_statistics(eval_path / f"predictions_{name}.txt", name, model.timer.get())
        model.timer.reset() # Reset the timer for the next split

        if args.overview_save_path != "":
            print("Saving overview to", args.overview_save_path)
            # Check if file exists, if not create it
            if not Path(args.overview_save_path).exists():
                with open(args.overview_save_path, "w") as f:
                    f.write("Evaluation Overview\n")
                    f.write("---------------------------------------------------------------------------------------------------------\n")
            if args.overview_run_name == "":
                args.overview_run_name = f"{args.corpus}/{model.LLM_model_type[0] if len(model.LLM_model_type) == 1 else 'Multi'}/{model.LLM_model_name[0] if len(model.LLM_model_name) == 1 else 'Multi'}/{timestamp}"
            with open(args.overview_save_path, "a") as f:
                f.write(args.overview_run_name + ": " + overview_string + "\n")

    print("Evaluation completed. Results saved to disk.")


def main(args, local_LLM=None):
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = LLMDualEncoderEntityDisambiguation.load(model_path=args.model_path,)
    print("Model loaded successfully.")

    # Load the Corpus
    print(f"Loading corpus {args.corpus}...")
    model_args = identify_specifications_from_model_identifier(Path(args.model_path).parent / "model_args.txt")
    if args.corpus == "AIDA":
        corpus = NEL_ENGLISH_AIDA(document_level=model_args["document_level"], use_ids_and_check_existence = True, wikipedia_user_agent="FlairAIDABot/1.0 (gamerfreakgoogl@gmail.com)")
        label_type = "nel"  # define under which "columns" the labels are found
    elif args.corpus == "ZELDA":
        corpus = ZELDA(document_level=model_args["document_level"])
        label_type = "nel"  # define under which "columns" the labels are found
    else:
        # See if specified corpus is subset of Zelda
        file_name = f"test_{args.corpus.lower()}.conll"
        # See if file exists
        if not Path(f"./ZELDA/test_data/conll/{file_name}").exists():
            raise NotImplementedError(f"Corpus {args.corpus.lower()} not found in ZELDA test data.")
        
        corpus = ColumnCorpus("ZELDA/test_data/conll/",
                            test_file=file_name,
                            column_format={0: "text", 1: "id", 2: "nel"} if args.corpus.lower() not in ["msnbc", "ace2004", "aquaint"] else {0: "text", 1: "nel"},
                            autofind_splits=False,
                            sample_missing_splits=False,
                            column_delimiter="\t",
                            document_level=model_args["document_level"],
                            document_separator_token="-DOCSTART-",
                            in_memory=True,
                            )
        
        label_type = "nel"  # define under which "columns" the labels are found
        args.skip_dev = True  # Skip dev set for these datasets
        args.corpus = args.corpus.lower()  # Set corpus name to lowercase for consistency


    #elif args.corpus in ["reddit-posts", "reddit-comments", "tweeki"]:
    #    args.skip_dev = True  # Skip dev set for these datasets
    #    corpus = NEL_ENGLISH_REDDIT()
    #    label_type = "nel"  # define under which "columns" the labels are found


    print("Corpus loaded successfully.")

    # Load local LLM if applicable
    if args.local_LLM and not local_LLM:
        local_LLM = LLM(model=args.LLM_model_name[0], max_model_len=30000)

    # ML Model variables
    model._label_type = label_type
    model.token_encoder.allow_long_sentences = True
    model.similarity_metric.metric_to_use = model_args["similarity_metric"]
    model.embedding_pooling = model_args["embedding_pooling"]

    
    # LLM variables
    model.LLM_model_type = args.LLM_model_type  # List of: Google, OpenAI, HU
    model.LLM_model_name = args.LLM_model_name # List of exact model names
    model.local_llm = local_LLM  # Whether to use a local LLM model. Default: False

    model.batching = args.batching  # Determine if the Batch API is used. Default: False
    model.regenerate_sentences = args.regenerate_sentences # Determines if sentence predictions are regenerated or loaded from disk. Default: False

    model.iterations = args.iterations # Number of LLM prediction rounds. Default: 3
    model.threshold = args.threshold # Threshold for LLM predictions. If threshold is set to 0.8, the bottom 20% of predictions are processed by the LLM.
    model.LLM_strategy = args.LLM_strategy # default, all # Determines whether to predict all tokens or only select ones
    model.LLM_selection_strategy = args.LLM_selection_strategy # similarity, difference, both # Determines which entities are selected as difficult
    model.LLM_verbalization_strategy = args.LLM_verbalization_strategy # description, categories, both, none # Determines how the candidates are verbalized for the LLM
    model.LLM_verbalization_candidate_strategy = args.LLM_verbalization_candidate_strategy # description, categories, both, none, same # Determines how the candidates are verbalized for the LLM candidate selection

    model.top_k = args.top_k # Number of top-k predictions to consider. Default: 5
    model.dynamic_top_k = args.dynamic_top_k  # Whether to use dynamic top-k based on the span score. Default: False

    model.reasoning = args.reasoning  # Reasoning level for LLM predictions. Must be stated if the model supports reasoning.

    model.num_agents = args.num_agents  # Number of LLM agents to use for LLM predictions. Default: 1

    model.timer = PausableTimer()  # Initialize the timer for LLM predictions

    if args.LLM_verbalization_strategy in ["categories", "description+categories"] or args.LLM_verbalization_candidate_strategy in ["categories", "description+categories"]:
        with Path(args.verbalization_path).open("r", encoding="utf-8") as fh:
            verbalization_data = json.load(fh)
        if not isinstance(verbalization_data, dict):
            raise ValueError(f"Expected top-level JSON object in {args.verbalization_path}")
        model.verbalization_data = verbalization_data
        print("Loaded extended label verbalization data")


    # Load API keys
    api_key = {}
    for i in range(args.num_agents):
        api_key[model.LLM_model_type[i]] = load_api_key(model.LLM_model_type[i])
    
        if not api_key[model.LLM_model_type[i]] and not model.LLM_model_type[i] in ["HU", "LOCAL"]:
            raise ValueError(f"API key for {model.LLM_model_type[i]} not found. Please set the environment variable for {model.LLM_model_type[i]} API key.")

    print(f"API keys loaded for {list(api_key.keys())}")
    model.api_key = api_key


    predict(model, corpus, args, model_args)


def sanity_check_and_set_arguments(args):
    # Check if num_agents matches the number of LLM model names
    # If you want the same model for each agent, you can provide a single model name instead of listing it n times
    if args.num_agents != len(args.LLM_model_name):
        if len(args.LLM_model_name) == 1:
            args.LLM_model_name = [args.LLM_model_name[0]] * args.num_agents
        else:
            raise ValueError(f"Number of agents ({args.num_agents}) does not match number of LLM model names ({len(args.LLM_model_name)}).")

    # Set LLM Model types
    model_types = []
    for i in range(args.num_agents):
        if "gemini" in args.LLM_model_name[i].lower():
            model_types.append("Google")
        elif "gpt" in args.LLM_model_name[i].lower() or args.LLM_model_name[i].lower().startswith("o"):
            model_types.append("OpenAI")
        elif "hu" in args.LLM_model_name[i].lower():
            model_types.append("HU")
        elif args.local_LLM:
            model_types.append("LOCAL")
        else:
            raise NotImplementedError(f"LLM Model type for {args.LLM_model_name} not recognized. Please set the LLM model type manually.")
    args.LLM_model_type = model_types
    
    # Set the reasoning parameter
    # For non reasoning models, we set it to 'none'
    reasonings = []
    for model in args.LLM_model_name:
        if "o4-mini" in model.lower() or "gpt-5" in model.lower():
            if args.reasoning == "none":
                raise ValueError(f"The Model {model} requires reasoning to be set. Please set the reasoning level to 'low', 'medium', or 'high'.")
            reasonings.append(args.reasoning)
        else:
            reasonings.append("none")
    args.reasoning = reasonings

    # Check if maximum output tokens are set correctly when reasoning is enabled
    if ("low" in args.reasoning or "medium" in args.reasoning or "high" in args.reasoning) and args.max_output_tokens < 1000:
        print("Warning: The maximum number of output tokens for the LLM is set to less than 1000. Exiting the program.")
        sys.exit(1)

    if args.threshold < 0 and "_abs" not in args.LLM_selection_strategy:
        args.LLM_selection_strategy += "_abs"

    if args.candidate_ordering == "random":
        args.random_candidate_order = True

    if args.LLM_verbalization_candidate_strategy == "same":
        args.LLM_verbalization_candidate_strategy = args.LLM_verbalization_strategy

    print("Sanity Check and Argument Fixes passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LLM Entity Disambiguation model.")
    parser.add_argument("--config", type=str, help="Path to a JSON config file containing all arguments.")
    parser.add_argument("--model_path", type=str, default="./Implementation/Models/final-model/final-model.pt", help="Path to the trained model file.")
    parser.add_argument("--corpus", type=str, default="AIDA", help="Name of the corpus to evaluate on.")
    parser.add_argument("--LLM_model_name", type=str, nargs="+", help="Name(s) of the LLM model to use.")
    parser.add_argument("--local_LLM", action="store_true", help="Whether to use a local LLM model.")
    parser.add_argument("--batching", action="store_true", help="Whether to use batching for LLM predictions.")
    parser.add_argument("--regenerate_sentences", action="store_true", help="Whether to regenerate sentences or load them from disk.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of LLM prediction iterations.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for LLM predictions. Higher values mean less LLM predictions.")
    parser.add_argument("--LLM_strategy", type=str, default="default", choices=["default", "all"], help="Strategy for LLM predictions.")
    parser.add_argument("--LLM_selection_strategy", type=str, default="similarity", choices=["similarity", "difference", "both", "similarity_abs", "difference_abs", "both_abs"], help="Selection strategy for LLM predictions.")
    parser.add_argument("--LLM_verbalization_strategy", type=str, default="description", choices=["description", "categories", "description+categories", "excerpt", "none"], help="Verbalization strategy for LLM predictions.")
    parser.add_argument("--LLM_verbalization_candidate_strategy", type=str, default="same", choices=["description", "categories", "description+categories", "excerpt", "none"], help="Verbalization strategy for LLM candidate predictions.")
    parser.add_argument("--verbalization_path", type=str, default=str("./verbalizations_structured.json"), help="Path to the verbalization JSON file.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top-k predictions to consider.")
    parser.add_argument("--dynamic_top_k", action="store_true", help="Whether to use dynamic top-k based on the span score.")
    parser.add_argument("--min_batch", type=int, default=-1, help="Skips all batches below minBatch. Set to -1 to evaluate all batches.")
    parser.add_argument("--max_batch", type=int, default=-1, help="Evaluates all batches up to maxBatch. Set to -1 to evaluate all batches.")
    parser.add_argument("--disable_LLM", action="store_true", help="Disables LLM predictions for this evaluation. Useful for testing the ML model without LLM predictions.")
    parser.add_argument("--rate_limit_timeout", type=int, default=10, help="Time to wait to avoid rate limiting for the LLM. Set to 0 to disable rate limiting.")
    parser.add_argument("--skip_dev", action="store_true", help="Skip evaluation on the dev set. Only evaluate on the test set.")
    parser.add_argument("--skip_test", action="store_true", help="Skip evaluation on the test set. Only evaluate on the dev set.")
    parser.add_argument("--max_output_tokens", type=int, default=200, help="Maximum number of output tokens for the LLM.")
    parser.add_argument("--spans_per_prompt", type=int, default=5, help="Number of spans per prompt for LLM predictions")
    parser.add_argument("--random_candidate_order", action="store_true", help="Randomize the order of candidates for LLM predictions.")
    parser.add_argument("--candidate_ordering", type=str, default="sorted", choices=["sorted", "random"], help="Ordering strategy for candidates in LLM predictions.")
    parser.add_argument("--reasoning", type=str, default="none", choices=["none", "low", "medium", "high"], help="Reasoning level for LLM predictions. Must be stated if the model supports reasoning.")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of LLM agents to use for LLM predictions. Required if more than one are to be used")
    parser.add_argument("--allow_extra_iteration", action="store_true", help="Allow an extra iteration for the LLM predictions for spans which could were not successfully predicted in the final iteration.")
    parser.add_argument("--overview_save_path", type=str, default="", help="Path to save the evaluation overview. If not set, it won't be saved.")
    parser.add_argument("--overview_run_name", type=str, default="", help="Name of the run to be included in the overview. If not set, corpus, model name and timestamp will be used.")

    args = parser.parse_args()

    # If config file is specified, override args with config values
    if args.config:
        with open(args.config, "r") as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            setattr(args, key, value)

    sanity_check_and_set_arguments(args)
    

    main(args)
