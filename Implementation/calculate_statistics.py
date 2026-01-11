from pathlib import Path
import pandas as pd
import json

def div(a, b):
    return f"{(a/b):.2%}" if b != 0 else "0.00%"

def calc_and_write_statistics(file_path, split = "test", time_taken = 0, print_only = False, custom_threshold=-1):

    if custom_threshold != -1:
        print_only = True

    #Load file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove all lines that don't start with '"'
    lines = [line for line in lines if line.startswith('"') and '---' in line]

    # Split each line in multiple lines at each occurence of '])"'
    # The lines list should not include lists but only lines
    parsed_lines = []
    for line in lines:
        parsed_lines.extend(line.split(')]"'))

    # Create empty pandas dataframe with columns "Gold", "Predict", "LLMPredict", "Top_k"
    # Also add columns for top_{i} and top_{i}_score for i in range(1, max_top_k + 1)

    # Get top_k
    parameter_file = Path(file_path).parent / "run_parameters.json"
    # Load dictionary from json file and extract value for key "top_k"
    try:
        with open(parameter_file, 'r') as f:    
            parameters = json.load(f)
        max_top_k = parameters.get("top_k", 5)  # Default to 5 if not found
        prompt_version = parameters.get("Prompt_Version", 1) # Default to 1 if not found
    except FileNotFoundError:
        print(f"Warning: {parameter_file} not found. Using default max_top_k = 5.")
        max_top_k = 5  # Default value if the file is not found
        prompt_version = 1

    column_names = ["Gold", "Predict", "LLMPredict", "Top_k"]
    for i in range(1, max_top_k + 1):
        column_names.append(f"Top_{i}")
        column_names.append(f"Top_{i}_score")
    column_names.append("LLM_Start")
    column_names.append("LLM_End")
    column_names.append("Input_Tokens")
    column_names.append("Output_Tokens")
    column_names.append("Span_Percentage")
    column_names.append("Top_k_Used")

    df = pd.DataFrame(columns=column_names)

    for idx, line in enumerate(parsed_lines):
        splits = line.split('---')

        gold = splits[0].split('/')[1].strip()
        pred = splits[1].strip()
        llm_pred = splits[2].strip()
        top_k = splits[3].strip()
        llm_iter = splits[4].strip() if len(splits) > 4 else '"N/A", "N/A"'
        input_tokens = int(splits[5].split(',')[0].strip()) if len(splits) > 5 else "0"
        output_tokens = int(splits[5].split(',')[1].strip()) if len(splits) > 5 else "0"
        span_percentage = int(splits[6].strip()) if len(splits) > 6 else -1
        top_k_used = int(splits[7].strip()) if len(splits) > 7 else max_top_k

        # If a custom percentage is provided and the data contains span_percentages set the llm_pred for
        # all spans below the span_percentage to -1
        if custom_threshold != -1 and span_percentage != -1:
            llm_pred = "N/A" if span_percentage < int(custom_threshold * 100) else llm_pred

        # Parse top_k values
        top_k_vals = {}
        top_k_splits = top_k.split('), (')
        top_k = "), (".join([top_k_splits[i] for i in range(top_k_used if top_k_used > 0 else max_top_k)]).strip()
        for i in range(0, max_top_k):
            if i >= top_k_used:
                top_k_vals[i+1] = ("N/A", 0.0)
                continue
            top_i_splits = top_k_splits[i].split(', ')

            # Remove quotes, whitespaces, commas, parenthesises and brackets from strings
            top_k_vals[i+1] = (top_i_splits[0].replace(",", "").replace('"', '').replace("'", "").replace("(", "").replace("[", "").strip(),
                            float(top_i_splits[1].replace(")", "").replace("]", "").strip()))

        
        column_fields = [gold, pred, llm_pred, str(top_k)]
        for i in range(1, max_top_k + 1):
            column_fields.append(top_k_vals[i][0])  # Top_i
            column_fields.append(top_k_vals[i][1])  # Top_i_score
        column_fields.append(llm_iter.split(',')[0].replace('"', '').strip())  # LLM_Start
        column_fields.append(llm_iter.split(',')[1].replace('"', '').strip())  # LLM_End
        column_fields.append(input_tokens)  # Input_Tokens
        column_fields.append(output_tokens)  # Output_Tokens
        column_fields.append(span_percentage)  # Span_Percentage
        column_fields.append(top_k_used)  # Top_k_Used

        df.loc[len(df)] = column_fields


    # Get number of lines in the dataframe
    num_lines = len(df)
    # Get number of lines where llm_predict != "N/A"
    num_llm_lines = len(df[df['LLMPredict'] != "N/A"])

    write_lines = []
    # Get number of lines where gold == predict
    num_correct = len(df[df['Gold'] == df['Predict']])
    write_lines.append(f"The Dual Encoder predicted correctly {num_correct} out of {num_lines} lines. ({div(num_correct, num_lines)})\n")
    # Get number of lines where gold included as a substring in top_k
    num_correct_in_top_k = len(df[df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1)])
    write_lines.append(f"    The Dual Encoder predicted correctly {num_correct} out of {num_correct_in_top_k} lines where the gold label was available in the top_k predictions. ({div(num_correct, num_correct_in_top_k)})\n")
    # Get number of lines where gold == predict and LLMPredict != "N/A"
    num_de_correct_and_llm_predicted = len(df[(df['Gold'] == df['Predict']) & (df['LLMPredict'] != "N/A")])
    write_lines.append(f"The Dual Encoder predicted correctly {num_de_correct_and_llm_predicted} out of {num_llm_lines} lines where the LLM also made a prediction (hard cases only). ({div(num_de_correct_and_llm_predicted, num_llm_lines)})\n")
    # Get the number of lines where llm_predict != "N/A" and gold is included as a substring in top_k
    num_llm_predicted_in_top_k = len(df[(df['LLMPredict'] != "N/A") & (df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1))])
    # Get the number of lines where gold == predict and llm_predict != "N/A" and gold is included as a substring in top_k
    num_de_correct_and_llm_predicted_in_top_k = len(df[(df['Gold'] == df['Predict']) & (df['LLMPredict'] != "N/A") & (df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1))])
    write_lines.append(f"    The Dual Encoder predicted correctly {num_de_correct_and_llm_predicted_in_top_k} out of {num_llm_predicted_in_top_k} lines where the LLM also made a prediction and the gold label was available in the top_k predictions. ({div(num_de_correct_and_llm_predicted_in_top_k, num_llm_predicted_in_top_k)})\n")

    write_lines.append("\n")
    # Get number of lines where gold == llm_predict and llm_predict != "N/A"
    num_llm_correct = len(df[(df['Gold'] == df['LLMPredict']) & (df['LLMPredict'] != "N/A")])
    write_lines.append(f"The LLM predicted correctly {num_llm_correct} out of {num_llm_lines} lines (hard cases only). ({div(num_llm_correct, num_llm_lines)})\n")
    # Get number of lines where the dual encoder is right and the llm_predict is not "N/A"
    num_de_correct_and_llm_predicted = len(df[(df['Gold'] == df['Predict']) & (df['LLMPredict'] != "N/A")])
    # Get number of lines where the dual encoder and llm_predict the same and are right
    num_llm_and_de_correct = len(
        df[(df['Gold'] == df['LLMPredict']) & (df['Gold'] == df['Predict']) & (df['LLMPredict'] != "N/A")]
    )
    write_lines.append(f"    The LLM predicted correctly {num_llm_and_de_correct} out of {num_de_correct_and_llm_predicted} lines where the Dual Encoder also predicted correctly. ({div(num_llm_and_de_correct, num_de_correct_and_llm_predicted)})\n")
    # Get number of lines where gold != llm_predict and gold != predict and llm_predict != "N/A" and gold is not included as a substring in top_k
    num_llm_correct_not_available = len(
        df[
            (df['LLMPredict'] != "N/A")
            & (df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1))
        ]
    )
    write_lines.append(f"    The LLM predicted correctly {num_llm_correct} out of {num_llm_correct_not_available} lines where the gold label was available in the top_k predictions. ({div(num_llm_correct, num_llm_correct_not_available)})\n")

    write_lines.append("\n")
    # Get number of lines where dual encoder != llm_predict and llm_predict != "N/A"
    num_differing = len(df[(df['LLMPredict'] != df['Predict']) & (df['LLMPredict'] != "N/A")])
    write_lines.append(f"The Dual Encoder and LLM predictions differ in {num_differing} out of {num_llm_lines} lines. ({div(num_differing, num_llm_lines)})\n")
    # Get number of lines where llm correct but dual encoder not
    num_llm_only_correct = len(df[(df['Gold'] == df['LLMPredict']) & (df['Gold'] != df['Predict'])])
    write_lines.append(f"    The LLM predicted correctly {num_llm_only_correct} lines where the Dual Encoder did not.\n")
    # Get number of lines where dual encoder correct but llm not
    num_de_only_correct = len(df[(df['Gold'] == df['Predict']) & (df['Gold'] != df['LLMPredict']) & (df['LLMPredict'] != "N/A")])
    write_lines.append(f"    The Dual Encoder predicted correctly {num_de_only_correct} lines where the LLM did not.\n")
    # Get number of lines where dual encoder and llm are different but are both wrong
    num_both_wrong = len(
        df[
            (df['Gold'] != df['Predict'])
            & (df['LLMPredict'] != df['Predict'])
            & (df['Gold'] != df['LLMPredict'])
            & (df['LLMPredict'] != "N/A")
        ]
    )
    write_lines.append(f"    The Dual Encoder and LLM are different but both wrong in {num_both_wrong} out of {num_llm_lines} lines.\n")
    # Get number of lines where dual encoder and llm are different but are both wrong and the gold label was in the top_k predictions
    num_both_wrong_in_top_k = len(
        df[		(df['Gold'] != df['Predict'])
            & (df['LLMPredict'] != df['Predict'])
            & (df['Gold'] != df['LLMPredict'])
            & (df['LLMPredict'] != "N/A")
            & (df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1))
        ]
    )
    write_lines.append(f"        Out of those {num_both_wrong} lines, the gold label was available in the top_k predictions in {num_both_wrong_in_top_k} lines.\n")

    write_lines.append("\n")
    # Get number of lines where the LLM predicted "None of the above"
    num_llm_none = len(df[df['LLMPredict'] == "None of the above"])
    write_lines.append(f"The LLM predicted 'None of the above' in {num_llm_none} lines.\n")
    # Get number of lines where the LLM predicted "None of the above" and the gold label was not in the top_k predictions
    num_llm_none_not_in_top_k = len(
        df[	(df['LLMPredict'] == "None of the above")
            & (~df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1))
        ]
    )
    write_lines.append(f"    The LLM predicted 'None of the above' in {num_llm_none_not_in_top_k} lines where the gold label was not in the top_k predictions, i.e. the LLM was correct\n")

    write_lines.append("\n")
    # Get number of lines where gold == llm_predict or where llm_predict == "None of the above" and gold is not in top_k
    num_llm_total_correct = len(
        df[	(df['Gold'] == df['LLMPredict'])
            | ((df['LLMPredict'] == "None of the above") & (~df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1)))
        ]
    )
    write_lines.append(f"The LLM either predicted the correct label or correctly predicted 'None of the above' in {num_llm_total_correct} out of {num_llm_lines} lines. ({div(num_llm_total_correct, num_llm_lines)})\n")

    write_lines.append("\n")
    # Get number of lines where gold == predict and llm_predict == 'N/A' + number of lines where llm_predict == gold
    num_de_plus_llm_total_correct = len(
        df[	((df['Gold'] == df['Predict'])
            & ((df['LLMPredict'] == "N/A"))
        | (df['Gold'] == df['LLMPredict']))
        ]
    )
    write_lines.append(f"Ultimately, the Dual Encoder + LLM correctly predicted {num_de_plus_llm_total_correct} out of {num_lines} lines. ({div(num_de_plus_llm_total_correct, num_lines)})\n")

    # Get number of lines where gold is in top_k predictions and LLM made a prediction
    num_gold_in_top_k = len(df[df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1)
                            & (df['LLMPredict'] != "N/A")])
    write_lines.append(f"    The maximum accuracy for the LLM is {num_gold_in_top_k} out of {num_llm_lines} lines. ({div(num_gold_in_top_k, num_llm_lines)})\n")

    # Calculate the maximum overall accuracy of Dual Encoder + LLM
    num_de_plus_llm_max_accuracy = len(
        df[	(df['Gold'] == df['Predict'])
            | ((df['LLMPredict'] != "N/A") & (df.apply(lambda row: f"'{row['Gold']}'" in row['Top_k'], axis=1)))
        ]
    )
    write_lines.append(f"    The maximum overall accuracy is {num_de_plus_llm_max_accuracy} out of {num_lines} lines. ({div(num_de_plus_llm_max_accuracy, num_lines)})\n")

    # Calculate total number of input and output tokens
    total_input_tokens = df['Input_Tokens'].sum()
    total_output_tokens = df['Output_Tokens'].sum()

    overview_string = f"[{split.upper()}]\t{div(num_llm_correct, num_llm_lines)} LLM ({div(num_gold_in_top_k, num_llm_lines)} Max) / {div(num_de_correct_and_llm_predicted, num_llm_lines)} DE (hard) / {div(num_llm_total_correct, num_llm_lines)} LLM+None / {div(num_de_plus_llm_total_correct, num_lines)} Overall ({div(num_de_plus_llm_max_accuracy, num_lines)} Max) / {div(num_correct, num_lines)} DE Only | I: {total_input_tokens} - O: {total_output_tokens} | {num_llm_lines} / {num_lines} spans prompted ({div(num_llm_lines, num_lines)}) | {time_taken} | v{prompt_version}"

    write_lines.append(f"\n\n{overview_string}")

    # Write the statistics to a file
    if not print_only:
        with open(str(file_path)[:-4] + "_statistics.txt", 'w') as f:
            for line in write_lines:
                f.write(line)

    # Print the file we just wrote
    print("---------------------------------------------------------------------------------------------------------")
    for line in write_lines:
        print(line.strip())
    print("---------------------------------------------------------------------------------------------------------\n")

    return overview_string


if __name__ == "__main__":
    # Pass command line arguments to main
    import sys
    file_path = sys.argv[1]
    split = sys.argv[2]
    time_taken = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    custom_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    calc_and_write_statistics(file_path, split, time_taken, False, custom_threshold)