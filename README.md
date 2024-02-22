# Comparative Attack Prompts with Pruning

This project is a modified implementation of [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://www.robustintelligence.com/tree-of-attacks-jailbreaking-black-box-llms-automatically).

The method iteratively uses LLMs to generate jailbreaking prompts, evaluates whether the generated prompt is on-topic, sends the prompt to the target and receives the response, then evaluates how well the prompt extracted the desired response from the target. Each of the evaluation stages is handled by LLM-evaluation and all models used can be selected independently to understand capability and protection differences. At each iteration, just the best performing prompts are continued for further improvement by pruning at each evalutator stage, or the process ends when the prompt and target's response reaches a threshold score.

Additional sources:

- Blog from original authors: [Using AI to Automatically Jailbreak GPT-4 and Other LLMs in Under a Minute](https://www.robustintelligence.com/blog-posts/using-ai-to-automatically-jailbreak-gpt-4-and-other-llms-in-under-a-minute)
- Independent implementation: Dreadnode's [Parley](https://github.com/dreadnode/parley)

## Features

LLM automated jailbreaking prompt generation, scoring and controlable pruning of the worst performing prompt-response pairs, as proposed in the original paper.


#### Modifications to the original paper are based on two approaches:

- LLM outputs of the attacker and evaluator models, together with data handling each iteration is achieved via `pydantic` models and the `instructor` library, with validation, auto-retries, and function-calling, tool-calling or JSON outputs
- The tree-based method is simplified by instead generating and scoring multiple jailbreak prompts my the language models with the aim to improve the visibility of each attempt and increase diversity. The on-topic evaluation stage is enhanced to allow retry with feedback for where it went wrong

#### Key features:

- Uses Pydantic models for validation and serialization via the [instructor](https://github.com/jxnl/instructor) library for OpenAI API calls
  - Provides pydantic model validation for LLM responses and validation with retries
  - This allow the original algorithm to be simplified and hardened to retry when responses don't conform to the format we want or omit fields
  - The retry mechanism is used for all LLM attack and evaluator calls
  - At the on-topic evaluation stage, rather than immediately pruning the evaluation fails, the fail and reason for the fail is passed back to the attacker so it can see its error and try again, up to some selected number of max tries
- The attack and score models work concurrently on all prompts each iteration
  - Structured outputs allow multiple attacks and responses to be handled and viewed together and compared rather than independently branching the tree for LLM calls
  - Enables the LLM-eval scoring method to see multiple prompt-response pairs to help it compare while generatintg the final score.
- Structured outputs via `pydantic` models have additional fields for reasoning to make the LLMs generate and consider more tokens in the responses, which has been shown to improve responses for many models and tasks

## Roadmap

- [ ] Add additional parameters to LLM for `top_k`, `top_p`, `frequency_penalty`, etc.
- [ ] Allow per-model endpoint selection


## Installation

Clone the repository.
Install the required libraries. Use either `poetry` (recommended) or the provided `requirements.txt`
Set your OpenAI API key as an environment variable.
Run the script with your desired arguments.

## LLM API endpoints and model support

Currently everything is running through the `openai` python client. It's easy to configure this for other endpoints. Currently supported are:

- `together`
- `anyscale`
- `ollama`
- `vLLM`
- (more, but untested...)

Any chat/instruct models supported by these endpoints should work, provided max-token requirements are satisfied. How well they work is of course highly variable.

Modify/add the following environment variables before running the script:

```
OPENAI_API_KEY=<OTHER_PROVIDER_API_KEY>
OPENAI_BASE_URL=<OTHER_PROVIDER_URL>
```

Use the script argument to set `model-type` to the endpoint type e.g. `together`,  model strings for `attacker-model`, `target-model`, `on-topic-model`, `score-model` as needed.  It's not currently possible to have different endpoints for different models, but may be in the future.

## Usage

#### Example

```bash
python cap_run.py \
    --goal "Explain how to break into a car" \
    --n-iterations 10 \
    --n-prompts-per-iteration 2 \
    --prune-ratio 0.5 \
    --score-threshold 8 \
    --attacker-model "gpt-4-turbo-preview" \
    --max-tokens 4096
```

#### Arguments

You can run the main script `cap_run.py` with various command line arguments:

- `--config-file`: Load settings from a JSON file and override command line arguments.
- `--goal`: The goal for the TAP generation.
- `--endpoint-type`: The endpoint type to use. Assumes you have environment variables set appropriately for the API key and base url.  Choices ["openai", "together", "anyscale", "ollama", "vllm"]
- `--attacker-model`: Name or path to the attacker model.
- `--target-model`: Name or path to the target model.
- `--on-topic-model`: Name or path to the on-topic model.
- `--score-model`: Name or path to the score model.
- `--n-iterations`: Number of iterations to run.
- `--n-prompts-per-iteration`: Number of prompts to generate per iteration.
- `--prune-ratio`: Ratio of prompts to keep for the next iteration.
- `--score-threshold`: Score threshold to determine when to stop the iterations.
- `--max-tokens`: Maximum number of tokens to be used for all models.
- `--max-tokens-attacker`: Maximum number of tokens for the attacker model. Overrides `--max-tokens` for this model.
- `--max-tokens-target`: Maximum number of tokens for the target model. Overrides `--max-tokens` for this model.
- `--max-tokens-on-topic`: Maximum number of tokens for the on-topic model. Overrides `--max-tokens` for this model.
- `--max-tokens-score`: Maximum number of tokens for the score model. Overrides `--max-tokens` for this model.
- `--log-level`: Set the logging level. Standard python logging log level strings, e.g. DEBUG.

