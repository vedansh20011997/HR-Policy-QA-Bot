# HR-Policy-QA-Bot
This is QA Bot made with the help of RAG architecture exploiting all the individual search components
<img width="1073" alt="image" src="https://github.com/user-attachments/assets/f3be4187-b6d2-4064-a316-25b41105a7c6">

Directory structure-<br>
|── README.md <br>
├── dockerfile <br>
├── inference.py <br>
├── main.py <br>
├── .env <br>
├── requirements.txt <br>
├── src <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── evaluation <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── evaluation_framework.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── ground_truth_creation.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── ground_truth_data.json <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── generator <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── generation.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── system_prompt_template.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── indexer <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── es_indexer.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── parser_and_chunk_creator <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── pdf_chunk_creator.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── reranker <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── reranker.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── retreiver <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp; ├── es_retreiver.py <br>
&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp; └── get_policy_name.py <br>
└── startup.sh <br>
___________________________________________________________________________________________________________________________________________________________________________________

How to start the QA fastapi rest API - 
1. git clone https://github.com/vedansh20011997/HR-Policy-QA-Bot.git
2. cp {path}/.env .
2. cd HR-Policy-QA-Bot
3. chmod +X startup.sh
4. bash ./startup.sh
___________________________________________________________________________________________________________________________________________________________________________________

How to manually start the api - 
$ python main.py
This runs the API at port 8000
___________________________________________________________________________________________________________________________________________________________________________________

How to run a sample request after the app is running - 
$ python inference.py
___________________________________________________________________________________________________________________________________________________________________________________

How to do indexing - 
1. Set the PDF_DIRECTORY and index_name parameter in the .env where the pdfs are located
2. cd src/indexer
3. $ python es_indexer.py
___________________________________________________________________________________________________________________________________________________________________________________

How to create the evaluation dataset - 
1. cd src/evaluation/
2. $ python ground_truth_creation.py
A sample of 100 ground truth samples are placed at - src/evaluation/ground_truth_data.json
___________________________________________________________________________________________________________________________________________________________________________________

All the individual scripts are capable of running on their own. 
Just move to the folder and run - 
$ python {script_name}.py
___________________________________________________________________________________________________________________________________________________________________________________

Application logs can be accessed at root directory - app.logs
___________________________________________________________________________________________________________________________________________________________________________________

Directories explaintion - 
1. src - this contains all the code<br>
&nbsp;&nbsp;&nbsp;&nbsp;a. src/parser_and_chunk_creator/pdf_chunk_creator.py - Code for parsing the pdf files and chunking them<br>
&nbsp;&nbsp;&nbsp;&nbsp;b. src/indexer/es_indexer.py - Code for consuming the chunks and indexing into ES according to the mapping defined.<br>
&nbsp;&nbsp;&nbsp;&nbsp;c. src/retreiver<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. get_policy_name.py - Code for assigning a policy type tag to the user_query enabling pre-filtering<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. es_retreiver.py - Code for fetching the relevant chunks for ES given a user_query. Five strategies are explored here<br>
&nbsp;&nbsp;&nbsp;&nbsp;d. src/reranker/reranker.py - Code for re-ranking the top-k documents received from retreiver<br>
&nbsp;&nbsp;&nbsp;&nbsp;e. src/generator/generation.py - Few shot generation using the context received from retreival + reranker<br>
&nbsp;&nbsp;&nbsp;&nbsp;f. src/evaluation<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. grouth_truth_creation.py - Code for generating the ground truth questions for random 100 chunks from ES<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. evaluation_framework.py - Based on ground truth data obtained, compare 20 statergies based on relevance and performance metrics<br>
 2. inference.py - Running this script posts a request to get running API at port 8000<br>
 3. main.py - main driver code that initiated and runs a fastapi server<br>
    swagger_url - http://localhost:8000/docs#/default/ask_question_api_ask_post<br>
 4. dockerfile<br>
 5. requirements.txt<br>
 6. startup.sh - shell script to run docker commands<br>
___________________________________________________________________________________________________________________________________________________________________________________

Evaluation results - <br>
<img width="612" alt="image" src="https://github.com/user-attachments/assets/e8a5e058-ce25-447e-9ed6-34977ad39aea">

