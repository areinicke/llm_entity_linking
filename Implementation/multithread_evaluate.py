import threading
import subprocess


def run_evaluation(cmd):
    """
    Function to run the evaluation command in a separate thread.
    """
    print(f"Running command: {cmd}")
    #subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, shell=True)

def main():
    num_threads = 5  # Number of threads to use
    min_batch_num = 0  # Minimum batch number to process
    max_batch_num = 10 # Maximum batch number to process (not inclusive)

    threads = []
    for i in range(num_threads):
        min_batch = min_batch_num + (i * (max_batch_num - min_batch_num) // num_threads)
        max_batch = min_batch_num + ((i + 1) * (max_batch_num - min_batch_num) // num_threads)
        if i == num_threads - 1:  # Last thread takes the remainder
            max_batch = max_batch_num

        cmd = f"python ./Implementation/evaluate.py --LLM_model_type HU --LLM_model_name HU-1 --corpus ZELDA --top_k 20 --skip_dev --disable_LLM --min_batch {min_batch} --max_batch {max_batch}"

        thread = threading.Thread(target=run_evaluation, args=(cmd,))
        thread.start()
        threads.append(thread)


    for idx, thread in enumerate(threads):
        thread.join()
        print(f"Thread {idx}/{num_threads} has finished execution.")

if __name__ == "__main__":
    main()