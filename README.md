# Spider

## GraphRAG

To run GraphRAG, follow the installation instruction [here](https://github.com/microsoft/graphrag). To run in CLI mode with custimized configuration,  you can install GraphRAG via Pip:

```
pip install graphrag
```

Then create an input folder to store your customized data.
```
mkdir -p ./GraphRAG/test/input
```
Then put some text files inside the input folder you just created. There can be multiple files or multiple folders containing multiple files.

### Data Collection
We provided a script `collect_data.py` to source articles from Wikipedia given a list of prompts. Save the prompts in the CSV format with the header "Prompt" and then change the corresponding path in the script to collect data. The script leverages [LangChain](https://python.langchain.com/docs/introduction/). You need to install the following the package before running the script:
```
pip install -qU langchain_community wikipedia
```

### Initialization
Run the following command to initialize a folder as GraphRAG's base:
```
graphrag init --root ./GraphRAG/test/input
```

Then two files will be generated for you in the `GraphRAG/test`: an environment file `.env` and a settings file `settings.yaml`.

You will need to specify your OpenAI API key in the `.env`: `GRAPHRAG_API_KEY=<YOUR_API_KEY>`. Additionally, you can change settings in `settings.yaml`. The most common one might be to change the driver language model to use. Inside `settings.yaml`, under the first `llm` block, change `model` to any OpenAI LLM model that fits your budget. For a complete list of models, check [here](https://openai.com/api/pricing/).

### Running the indexing of the tree
To form a hierarchical tree out of the text corpus you just stored, run:
```
graphrag index --root ./GraphRAG/test
```
A complete list of arguments can be found [here](https://microsoft.github.io/graphrag/cli/)

If you added new files in the input corpus, you can run:
```
graphrag update --root ./GraphRAG/test
```
to update the graph.

### Querying the knowledge graph with prompts
For a global understanding of the knowledge graph you just built, run:
```
graphrag query --root ./GraphRAG/test --method global --query "<YOUR QUESTION HERE>"
```
Similarly, to get an answer from local details, run:
```
graphrag query --root ./GraphRAG/test --method local --query "<YOUR QUESTION HERE>"
```
