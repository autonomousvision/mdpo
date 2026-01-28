import multiprocessing
import time
import sys
import os

# Critical: Disable tokenizer parallelism to prevent deadlocks when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset

# Timeout for each problem execution in seconds
TIMEOUT = 30


def unsafe_execute(code, result_queue):
    """
    Worker function to execute code in a separate process.
    """
    try:
        # We use a shared global dictionary to execute the code
        # This simulates a module environment
        exec_globals = {}
        exec(code, exec_globals)
        result_queue.put("passed")
    except AssertionError:
        result_queue.put("failed")
    except Exception as e:
        # Captures syntax errors, runtime errors, etc.
        result_queue.put(f"error: {str(e)}")


def check_correctness(generated_code, test_cases, timeout=TIMEOUT):
    """
    Evaluates a single problem by running the generated code against test cases.
    """
    # Construct the full script: Generated Code + Test Assertions
    # MBPP test cases are usually list of strings like "assert func(1)==2"
    full_script = generated_code + "\n\n" + "\n".join(test_cases)

    # Use 'spawn' context for safer process creation
    # This avoids 'fork' related threading deadlocks common with ML libraries
    ctx = multiprocessing.get_context("spawn")

    # Queue to communicate between processes
    queue = ctx.Queue()

    # Create a separate process for execution
    # This prevents infinite loops or segfaults from crashing the main script
    p = ctx.Process(target=unsafe_execute, args=(full_script, queue))
    p.start()

    # Wait for the process to finish or timeout
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return 0  # Fail due to timeout

    if not queue.empty():
        result = queue.get()
        return 1 if result == "passed" else 0
    else:
        return 0  # Fail due to unknown crash


def evaluate_mbpp(generated_solutions):
    """
    Calculates the accuracy of generated solutions against the MBPP dataset.

    Args:
        generated_solutions (list[str]): A list of Python code strings.
                                         Must align with the MBPP 'sanitized' test set order.

    Returns:
        float: The accuracy percentage (0.0 to 1.0).
    """

    # Load the MBPP dataset (sanitized version is standard for evaluation)
    # Ensure you have run: pip install datasets
    print("Loading MBPP dataset...")
    try:
        dataset = load_dataset("mbpp", "sanitized", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0.0

    # Validation: Ensure input length matches dataset length
    if len(generated_solutions) != len(dataset):
        print(f"Warning: Number of generated solutions ({len(generated_solutions)}) "
              f"does not match dataset size ({len(dataset)}). Evaluator will truncate or fail.")
        # For safety, we zip and handle the minimum length

    passed_count = 0
    total_count = len(dataset)

    print(f"Starting evaluation of {total_count} problems...")

    for i, (problem, code) in enumerate(zip(dataset, generated_solutions)):
        test_list = problem['test_list']

        # Check correctness
        is_correct = check_correctness(code, test_list)
        passed_count += is_correct

        # Optional: Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_count}...")

    accuracy = passed_count / total_count
    print(f"Final Accuracy: {accuracy:.2%}")

    return accuracy


# --- Example Usage ---
if __name__ == "__main__":
    # This is a dummy example to demonstrate how to call the function.
    # In a real scenario, you would pass your model's outputs here.

    # Let's verify the first problem of MBPP (Task ID 111 usually in full set, varies in sanitized)
    # We will simulate a correct solution and an incorrect one.

    # Mocking what the dataset looks like for testing this script
    # (The script actually loads the real dataset, but we need generated code to feed it)
    print("Note: To run this fully, you need the 'datasets' library installed.")

    # If you wanted to test this script without generating 100+ solutions,
    # you would generate them using your LLM first.
    # example_generations = [model.generate(prompt) for prompt in prompts]
    # acc = evaluate_mbpp(example_generations)