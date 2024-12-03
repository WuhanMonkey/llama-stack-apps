## DocQA

This is an end-to-end Retrieval Augmented Generation (RAG) App leveraging llama-stack that handles the logic for ingesting documents, storing them in a vector database and providing an inference interface.

We share the details of how to run first and then an outline of how it works:

### Prerequisite:

Install docker: Check [this doc for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [this doc for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and this [instruction for Linux](https://docs.docker.com/engine/install/).

For Mac and Windows users, you need to start the Docker app manually after installation.

### How to run the pipeline:

![RAG_workflow](./data/assets/DocQA.png)

The above is the workflow diagram for this RAG app. To run the app, please read the following instructions:

1.

```bash
pip install chromadb
chroma run --host localhost --port 6000 --path ./example_data &
ollama run llama3.2:1b-instruct-fp16 --keepalive=24h &
python app.py
```
