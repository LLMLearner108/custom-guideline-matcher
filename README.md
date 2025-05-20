# custom-guideline-matcher

To use this, create a .env file in the root directory and mention the OpenAI key here

```bash
OPENAI_API_KEY="sk-1kdah321"
```

Then you can set the constant values in the `guideline_proposer_standalone.py` script based on your configuration and run the script directly to get the results.

## Points to remember

1. Here we assume that the user messages start with `user:` and the assistant messages start with `assistant:`. If it is not the case, then please make changes accordingly in the function `conversation_to_events`
2. Feel free to change the batching logic by making modifications in `CustomGuidelineMatching._get_optimal_batch_size` within the `overriden_guideline_matcher.py` file. Right now it is set to 3 batches in a row.