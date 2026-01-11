from vllm import LLM, SamplingParams
import os

system_msg = """
You are a professional entity-disambiguation annotator. For each question in the prompt you must select exactly one answer number.
Hard rules (must follow):
1. Read the full text, then answer every question in order (Q1, Q2, ...).
2. For each question, consider ONLY the mention marked with that question tag (e.g., [Q1]) and ONLY the options listed immediately after that question. Do NOT reuse options from other questions.
3. Choose the option that best matches the mention given the context. Prefer contextual relevance over surface-form match when they conflict. Prefer the most specific, contextually accurate entry when multiple options fit.
4. If no option matches, choose '0' (None of the above).
5. Sometimes, the answer numbering may not be continuous. Carefully look at the answer number of your selected option and provide it.
6. Output format: Exactly N lines for N questions. Each line must contain EXACTLY one digit (the chosen option number). The first line is the answer for Q1, the second for Q2, etc. No other characters, labels, punctuation, or explanation. No blank lines.
7. Do NOT provide chain-of-thought or any additional text in your answer.

-------------------------------------------------

"""
prompts = ["""
Consider the following text:

CRICKET - LARA (West Indian cricketer) ENDURES ANOTHER MISERABLE DAY . Robert Galvin MELBOURNE [**Q1**] 1996-12-06 Australia (national sports team) gave Brian Lara (West Indian cricketer) another reason to be miserable when they beat West Indies (multinational cricket team) by five wickets in the opening World Series [**Q2**] limited overs match on Friday . Lara (West Indian cricketer) , disciplined for misconduct on Wednesday , was dismissed for five to extend a disappointing run of form on tour . Australia (national sports team) , who hold a 2-0 lead in the five-match test series , overhauled West Indies (multinational cricket team) ' total of 172 all out with eight balls to spare to end a run of six successive one-day defeats . All-rounder Greg Blewett (Australian cricketer) steered his side to a comfortable victory with an unbeaten 57 in 90 balls to the delight of the 42,442 crowd . Man-of-the match Blewett (Australian cricketer) came to the wicket with the total on 70 for two and hit three fours during an untroubled innings lasting 129 minutes . His crucial fifth-wicket partnership with fellow all-rounder Stuart Law (Australian cricketer) , who scored 21 , added 71 off 85 balls . Lara (West Indian cricketer) looked out of touch during his brief stay at the crease before chipping a simple catch to Shane Warne (Australian cricketer) at mid-wicket . West Indies (multinational cricket team) tour manager Clive Lloyd (former World Cup winning captain of West Indies in 1975 and 1979) has apologised for Lara (West Indian cricketer) 's behaviour on Tuesday . He ( Lara (West Indian cricketer) ) had told Australia (national sports team) coach Geoff Marsh (cricketer) that wicketkeeper Ian Healy (Australian cricketer) was unwelcome in the visitors ' dressing room . The Melbourne (capital city of Victoria, Australia) crowd were clearly angered by the incident , loudly jeering the West Indies (multinational cricket team) vice-captain as he walked to the middle . It was left to fellow left-hander Shivnarine Chanderpaul (West Indian cricketer) to hold the innings together with a gritty 54 despite the handicap of an injured groin . Chanderpaul (West Indian cricketer) was forced to rely on a runner for most of his innings after hurting himself as he scurried back to his crease to avoid being run out . Pakistan (national sports team) , who arrive in Australia (country in Oceania) later this month , are the other team competing in the World Series [**Q3**] tournament .

----------

Which of the following entries does the mention 'MELBOURNE' indicated with [**Q1**] in the text refer to? Please provide the most relevant entry based on the context of the sentence and only respond with the number of the answer. Make sure to only consider the correct mention indicated with [**Q1**].

(1) Melbourne Cricket Ground - (Sports stadium in Melbourne)
(2) Melbourne - (capital city of Victoria, Australia)
(3) Melbourne Cricket Club - (sports club in Melbourne, Australia)
(4) Whitten Oval - (stadium in Melbourne, Victoria, Australia)
(5) Australian Tri-Series - (television series)
(0) None of the above or Unsure

Which of the following entries does the mention 'World Series' indicated with [**Q2**] in the text refer to? Please provide the most relevant entry based on the context of the sentence and only respond with the number of the answer. Make sure to only consider the correct mention indicated with [**Q2**].

(1) 1996 Cricket World Cup - (Cricket World Cup)
(2) World Series Cricket - (former cricket competition)
(3) West Indian cricket team in Australia in 1996–97 - (international cricket tour)
(4) 1996 World Series - (92nd edition of Major League Baseball's championship series)
(5) Australian cricket team in the West Indies in 1998–99 - (Australian cricket team in the West Indies in 1998–99)
(0) None of the above or Unsure

Which of the following entries does the mention 'World Series' indicated with [**Q3**] in the text refer to? Please provide the most relevant entry based on the context of the sentence and only respond with the number of the answer. Make sure to only consider the correct mention indicated with [**Q3**].

(1) World Series Cricket - (former cricket competition)
(2) World Series - (Championship of Major League Baseball)
(3) Australian cricket team in the West Indies in 1998–99 - (Australian cricket team in the West Indies in 1998–99)
(4) World Series Cricket West Indies XI - (Cricket team)
(5) World Series Hockey - (hockey tournament initiated by Indian Hockey Federation in 2012)
(0) None of the above or Unsure
"""]

messages_list = [
    [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
     for prompt in prompts
]

#os.environ["LLM_TEST_FORCE_FP8_MARLIN"] = "1"
#os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3000)

#llm = LLM(model="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", max_model_len=20000, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=2)
llm = LLM(model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16", max_model_len=20000, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=2)

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Generated text:\n{generated_text!r}")

# Using chat interface.
outputs = llm.chat(messages_list, sampling_params)
for idx, output in enumerate(outputs):
    print(f"Output:\n{output}")
    print("---------------------------------------------------------------------------------------------")
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Generated text:\n{generated_text!r}")

    # Check if </think> tags is present. If yes, strip everything before it
    if "</think>" in generated_text:
        cleaned_output = generated_text.split("</think>")[-1].strip()
        print(f"Cleaned Generated text:\n{cleaned_output!r}")
        print(f"Prompt Tokens: {len(output.prompt_token_ids)}")
        print(f"Response Tokens: {len(output.outputs[0].token_ids)}")