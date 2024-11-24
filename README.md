# HR-Policy-QA-Bot
This is QA Bot made with the help of RAG architecture exploiting all the individual search components

Directory structure-
── README.md <br>
├── dockerfile <br>
├── inference.py <br>
├── main.py <br>
├── .env <br>
├── requirements.txt <br>
├── src <br>
│   ├── evaluation <br>
│   │   ├── evaluation_framework.py <br>
│   │   ├── ground_truth_creation.py <br>
│   │   └── ground_truth_data.json <br>
│   ├── generator <br>
│   │   ├── generation.py <br>
│   │   └── system_prompt_template.py <br>
│   ├── indexer <br>
│   │   └── es_indexer.py <br>
│   ├── parser_and_chunk_creator <br>
│   │   └── pdf_chunk_creator.py <br>
│   ├── reranker <br>
│   │   └── reranker.py <br>
│   └── retreiver <br>
│       ├── es_retreiver.py <br>
│       └── get_policy_name.py <br>
└── startup.sh <br>

How to start the QA fastapi rest API - 
1. git clone https://github.com/vedansh20011997/HR-Policy-QA-Bot.git
2. cp {path}/.env .
2. cd HR-Policy-QA-Bot
3. chmod +X startup.sh
4. bash ./startup.sh

How to manually start the api - 
$ python main.py
This runs the API at port 8000

How to run a sample request after the app is running - 
$ python inference.py

How to do indexing - 
1. Set the PDF_DIRECTORY and index_name parameter in the .env where the pdfs are located
2. cd src/indexer
3. $ python es_indexer.py

How to create the evaluation dataset - 
1. cd src/evaluation/
2. $ python ground_truth_creation.py
A sample of 100 ground truth samples are placed at - src/evaluation/ground_truth_data.json

All the individual scripts are capable of running on their own. 
Just move to the folder and run - 
$ python {script_name}.py

Application logs can be accessed at root directory - app.logs
