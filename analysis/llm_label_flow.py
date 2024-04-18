from typing import List, Callable, Tuple, Any, Dict
from tqdm import tqdm
from threading import Thread, Lock, BoundedSemaphore
import time
import ast
import os
import queue
import openai
import google.cloud.aiplatform as aiplatform
import google.generativeai as palm
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.language_models import TextGenerationModel
from langchain.llms import GooglePalm
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

## Credential Set Up

# Set up Project Creds in GCP and Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials_gpe-analytics.json"
pid = 'gpe-analytics'
aiplatform.init(project=pid)

# Setup for PALM
import google.generativeai as palm
PALM_API_NEW_KEY = 'AIzaSyBs8XrLbmbAGIbFxcmPqGWXbWHCIyOj8MY'
palm.configure(api_key=PALM_API_NEW_KEY)

# Set up Project Creds in OpenAI here if planning to use ChatGPT
openai.api_type = "type"
openai.api_base = "type_base"
openai.api_version = "api_version"
openai.api_key="api_key"

# Set Up PaLM API KEY if planning to use the PaLM API
PALM_API_KEY = 'your_palm_api_key'

# Code:

class RateLimitedTaskExecutor:
    """Class to manage rate-limited task execution across multiple threads.

    Attributes:
        rate_limit (int): The maximum number of requests per second.
        worker_func (Callable): The function to be executed in each worker thread.
    """

    def __init__(self, rate_limit: int, worker_func: Callable):
        """Initialize the rate-limited task executor.

        Args:
            rate_limit (int): The maximum number of requests per second.
            worker_func (Callable): The function to be executed in each worker thread.
        """
        self.semaphore = BoundedSemaphore(rate_limit)
        self.rate_limit = rate_limit
        self.request_interval = 1 / rate_limit
        self.task_queue = queue.Queue()
        self.result_list = []
        self.lock = Lock()
        self.worker_func = worker_func

    def worker(self):
        """The worker thread that processes tasks from the queue."""
        while True:
            args = self.task_queue.get()
            if args is None:  # Sentinel value to exit worker
                break
            self.semaphore.acquire()
            start_time = time.time()
            result = self.worker_func(*args)
            with self.lock:
                self.result_list.append(result)
            elapsed_time = time.time() - start_time
            self.semaphore.release()

            sleep_time = self.request_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def execute(self, args_list: List[Tuple]) -> List[Any]:
        """Execute tasks with rate limiting.

        Args:
            args_list (List[Tuple]): List of argument tuples for the worker function.

        Returns:
            List[Any]: List of results from executing the worker function.
        """
        # Start worker threads
        workers = []
        for _ in range(self.rate_limit):
            t = Thread(target=self.worker)
            t.start()
            workers.append(t)

        # Enqueue tasks with pacing
        for args in tqdm(args_list):
            self.task_queue.put(args)
            time.sleep(self.request_interval)

        # Add sentinel values to signal workers to exit
        for _ in range(self.rate_limit):
            self.task_queue.put(None)

        # Wait for all worker threads to finish
        for t in workers:
            t.join()

        return self.result_list


class llm_component:
    """Class to encapsulate a flexible predictor with selectable output parsing models and text formats.

    Attributes:
        model_type (Type[BaseModel]): The Pydantic BaseModel to use for output parsing.
        llm (VertexAI): The language model for prediction.
        query (str): The query string to be used for generating prompts.
        prompt (PromptTemplate): Template for generating language model prompts.
        parser (PydanticOutputParser): Parser for parsing the output of the language model.
        parser_fixer (OutputFixingParser): Fallback parser to fix the output in case of exceptions.
    """

    def __init__(self, query: str, api_type: str, labels: list, rate_limit: int = 1):
        """Initialize the llmNLP class.

        Args:
            model_name (str): Name of the model for mapping to a Pydantic BaseModel.
            query (str): The query string.
            rate_limit (int): Number of API calls per second allowed by the LLM API service
        """
        # Labels
        self.labels = labels
        # Initialize the query string
        self.query = query
        # Initialize API
        self.api_type = api_type
        # Set the ratelimit in API calls per second
        self.rate_limit = rate_limit
        # Initialize the Language Model of your Choice
        self.llm = self.obtain_api_model()

        # Initialize text bison model for Langchain bypass for test
        self.text_bison = TextGenerationModel.from_pretrained("text-bison@001")
        self.gemini = GenerativeModel("gemini-pro")
        self.palm_model_id = 'models/text-bison-safety-off'

        # Set Text Bison Parameters

        # OG SETTING

        self.text_bison_parameters = {

            "max_output_tokens":100,
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 40
        }

        self.gemini_parameters =  {
        "max_output_tokens": 3000,
        "temperature": 0,
        "top_p": 0.8}





        # Get the prompt template based on the model type
        self.prompt = self.get_prompt_template()
        # Initialize the LLMChain object
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)


    def obtain_api_model(self):
        """Obtains and returns an instance of the API model based on the configured `api_type`.

            This function creates an instance of the specified API model using predefined parameters and credentials set in the environment (lines 14-17).

            Returns:
                An instance of the API model based on `api_type`:
                    - VertexAI with model 'text-bison@001' and temperature 0.0 if `api_type` is 'vertex-api'.
                    - GooglePalm with temperature 0.0 and Google API key `PALM_API_KEY` if `api_type` is 'palm-api'.
                    - OpenAI with temperature 0.0, OpenAI API key `OPENAI_API_KEY`, and model name 'gpt-4' if `api_type` is 'openai-api'.


            """

        # Note credentials are already set above in environment. Lines 14-17
        if self.api_type == 'vertex-api':
            return VertexAI(model='text-bison@001', temperature=0.0)

        if self.api_type == 'palm-api':
            return GooglePalm(temperature=0.0, google_api_key=PALM_API_KEY)

        if self.api_type=='openai-api':
            return OpenAI(temperature=0.0,openai_api_key=OPENAI_API_KEY,model_name='gpt-4')

    def get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for generating language model prompts.

        Returns:
            PromptTemplate: The prompt template.
        """

        prompt = PromptTemplate(
            template="{query}" + "Label the Text. \Text: {text}",
            input_variables=["query", "text"])
        return prompt

    def output_parser(self, output):
        """Parses the given output and returns a dictionary with a label key and corresponding value.

           The method converts the output to lowercase and checks if it contains any predefined labels. If a label is found, it returns a dictionary with the label as the value. If the output is empty, it returns a dictionary with 'cannot_convert' as the value. If no predefined label is found and the output is not empty, it returns a dictionary with 'unknown' as the value.

           Args:
               output (str): The output string to be parsed.

           Returns:
               dict: A dictionary with the key 'label' and the corresponding value based on the parsing logic described above.
           """
        output_lower = output.lower()
        for label in self.labels:
            if label in output_lower:
                return {'label': label}
        if output_lower == '':
            return {'label': 'cannot_convert'}
        return {'label': 'unknown'}  # In case no label matches and output is not empty

    def run_langchain(self, uid: str, text: str) -> Dict[str, Any]:
        """Run the prediction chain to generate the output.

        Args:
            uid (str): Unique identifier.
            text (str): Input text.

        Returns:
            Dict[str, Any]: Parsed output along with the unique identifier.
        """
        # Run the prediction chain
        output = self.llm_chain.run(query=self.query, text=text)
        parsed_output = self.output_parser(output)
        # Add the unique identifier to the parsed output
        parsed_output['uid'] = uid
        return parsed_output

    def run_text_bison(self,uid,text):
        """Generates a prompt using the query and text, predicts the label using the text_bison model,
    and then parses the output to return a dictionary with the label and unique identifier.

    The method concatenates the query and text to form a prompt, which is then passed to the text_bison model
    to predict the label. The predicted output is then parsed using the output_parser method to obtain the label.
    A unique identifier is added to the parsed output before returning the final result.

    Args:
        text (str): The text to be labeled.
        uid (str): The unique identifier to be added to the output.

    Returns:
        dict: A dictionary containing the label and unique identifier.
    """
        prompt = self.query + f""" Label the Text. \Text:'{text}' """
        output = self.text_bison.predict(prompt, **self.text_bison_parameters).text
        parsed_output = self.output_parser(output)

        # Add the unique identifier to the parsed output
        parsed_output['uid'] = uid
        return parsed_output

    def run_gemini_batch(self, batch,uids, texts):
        """Generates a prompt using the query and text, predicts the label using the text_bison model,
    and then parses the output to return a dictionary with the label and unique identifier.

    The method concatenates the query and text to form a prompt, which is then passed to the text_bison model
    to predict the label. The predicted output is then parsed using the output_parser method to obtain the label.
    A unique identifier is added to the parsed output before returning the final result.

    Args:
        text (str): The text to be labeled.
        uid (str): The unique identifier to be added to the output.

    Returns:
        dict: A dictionary containing the label and unique identifier.
    """
        input_dict = {uid:text for uid,text in zip(uids,texts)}
        prompt = self.query + f"""{input_dict}"""
        # prompt = self.query + f"""{texts}"""
        output = self.gemini.generate_content(prompt).text.replace('`','')
        parsed_output = {'output':output}
        # Add the unique identifier to the parsed output
        parsed_output['uids'] = uids
        parsed_output['texts'] = texts
        parsed_output['batch'] = batch

        return parsed_output

    def run_gemini(self,uid, text):
        """Generates a prompt using the query and text, predicts the label using the text_bison model,
    and then parses the output to return a dictionary with the label and unique identifier.

    The method concatenates the query and text to form a prompt, which is then passed to the text_bison model
    to predict the label. The predicted output is then parsed using the output_parser method to obtain the label.
    A unique identifier is added to the parsed output before returning the final result.

    Args:
        text (str): The text to be labeled.
        uid (str): The unique identifier to be added to the output.

    Returns:
        dict: A dictionary containing the label and unique identifier.
    """
        prompt = self.query + f""" Label the Text. \Text:'{text}' """
        try:

            output =  self.gemini.generate_content(prompt).text.replace('`','')
        except:
            output = ''
        parsed_output = self.output_parser(output)
        parsed_output['uid'] = uid

        return parsed_output

    def run_text_bison_context(self,uid,text):
        """Generates a prompt using the query and text, predicts the label using the text_bison model,
    and then parses the output to return a dictionary with the label and unique identifier.

    The method concatenates the query and text to form a prompt, which is then passed to the text_bison model
    to predict the label. The predicted output is then parsed using the output_parser method to obtain the label.
    A unique identifier is added to the parsed output before returning the final result.

    Args:
        text (str): The text to be labeled.
        uid (str): The unique identifier to be added to the output.

    Returns:
        dict: A dictionary containing the label and unique identifier.
    """
        prompt = self.query + f""" Label the Text. \Text:'{text}' """
        output = self.text_bison.predict(prompt, **self.text_bison_parameters).text
        parsed_output = self.output_parser(output)

        context_prompt = f"Given the rules listed below. Explain why the text was given the label in detail. \n\n{self.query} \n\nLabel: {parsed_output['label']} \n\nText: {text}"
        context_params = {

            "max_output_tokens": 250,
            "temperature": 0.99,
            "top_p": 0.8,
            "top_k": 40
        }
        context_output = self.text_bison.predict(context_prompt, **context_params).text
        parsed_output['context'] = context_output


        # Add the unique identifier to the parsed output
        parsed_output['uid'] = uid
        return parsed_output

    def run_text_bison_palm(self,uid,text):
        """Generates a prompt using the query and text, predicts the label using the text_bison model,
    and then parses the output to return a dictionary with the label and unique identifier.

    The method concatenates the query and text to form a prompt, which is then passed to the text_bison model
    to predict the label. The predicted output is then parsed using the output_parser method to obtain the label.
    A unique identifier is added to the parsed output before returning the final result.

    Args:
        text (str): The text to be labeled.
        uid (str): The unique identifier to be added to the output.

    Returns:
        dict: A dictionary containing the label and unique identifier.
    """
        prompt = self.query + f""" Label the Text. \Text:'{text}' """
        output = palm.generate_text(
            model=self.palm_model_id,
            prompt=prompt,
            temperature=0,
            max_output_tokens=200,
        )
        parsed_output = self.output_parser(output.result)

        # Add the unique identifier to the parsed output
        parsed_output['uid'] = uid
        return parsed_output

    def generate_labels_gemini_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels for a dataset.

        Checks if 'uid' and 'text' keys are in the data dictionary.
        If they are not, raises a KeyError.

        Args:
            data (Dict[str, Any]): The dataset containing 'uid' and 'text'.

        Returns:
            Dict[str, Any]: Results after running the prediction chain.

        Raises:
            KeyError: If either 'uid' or 'text' is not in data.
        """
        # Check if 'uid' and 'text' keys exist in data
        # if 'uids' not in data.keys() or 'texts' not in data.keys():
        #     raise KeyError("Both 'uids' and 'texts' keys must be present in the data dictionary.")

        # Initialize the RateLimitedTaskExecutor with a rate limit of 20 requests per second
        task_executor = RateLimitedTaskExecutor(rate_limit=self.rate_limit, worker_func=self.run_gemini)
        # Prepare the argument list for labeling
        args_list = [(batch['batch'],batch['uids'], batch['texts']) for batch in data]
        # Execute the labeling tasks
        results = task_executor.execute(args_list)

        return results

    def generate_labels_gemini(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels for a dataset.

        Checks if 'uid' and 'text' keys are in the data dictionary.
        If they are not, raises a KeyError.

        Args:
            data (Dict[str, Any]): The dataset containing 'uid' and 'text'.

        Returns:
            Dict[str, Any]: Results after running the prediction chain.

        Raises:
            KeyError: If either 'uid' or 'text' is not in data.
        """
        # Check if 'uid' and 'text' keys exist in data
        if 'uid' not in data.keys() or 'text' not in data.keys():
            raise KeyError("Both 'uid' and 'text' keys must be present in the data dictionary.")

        # Initialize the RateLimitedTaskExecutor with a rate limit of 20 requests per second
        task_executor = RateLimitedTaskExecutor(rate_limit=10, worker_func=self.run_gemini)
        # Prepare the argument list for labeling
        args_list = [(uid, text) for uid, text in zip(data['uid'], data['text'])]
        # Execute the labeling tasks
        results = task_executor.execute(args_list)


        return results


    def generate_labels_langchain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels for a dataset.

        Checks if 'uid' and 'text' keys are in the data dictionary.
        If they are not, raises a KeyError.

        Args:
            data (Dict[str, Any]): The dataset containing 'uid' and 'text'.

        Returns:
            Dict[str, Any]: Results after running the prediction chain.

        Raises:
            KeyError: If either 'uid' or 'text' is not in data.
        """
        # Check if 'uid' and 'text' keys exist in data
        if 'uid' not in data.keys() or 'text' not in data.keys():
            raise KeyError("Both 'uid' and 'text' keys must be present in the data dictionary.")

        # Initialize the RateLimitedTaskExecutor with a rate limit of 20 requests per second
        task_executor = RateLimitedTaskExecutor(rate_limit=self.rate_limit, worker_func=self.run_langchain)
        # Prepare the argument list for labeling
        args_list = [(uid, text) for uid, text in zip(data['uid'], data['text'])]
        # Execute the labeling tasks
        results = task_executor.execute(args_list)

        return results

    def generate_labels_text_bison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels for a dataset.

        Checks if 'uid' and 'text' keys are in the data dictionary.
        If they are not, raises a KeyError.

        Args:
            data (Dict[str, Any]): The dataset containing 'uid' and 'text'.

        Returns:
            Dict[str, Any]: Results after running the prediction chain.

        Raises:
            KeyError: If either 'uid' or 'text' is not in data.
        """
        # Check if 'uid' and 'text' keys exist in data
        if 'uid' not in data.keys() or 'text' not in data.keys():
            raise KeyError("Both 'uid' and 'text' keys must be present in the data dictionary.")

        # Initialize the RateLimitedTaskExecutor with a rate limit of 20 requests per second
        task_executor = RateLimitedTaskExecutor(rate_limit=self.rate_limit, worker_func=self.run_text_bison)
        # Prepare the argument list for labeling
        args_list = [(uid, text) for uid, text in zip(data['uid'], data['text'])]
        # Execute the labeling tasks
        results = task_executor.execute(args_list)

        return results

    def generate_labels_text_bison_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labels for a dataset.

        Checks if 'uid' and 'text' keys are in the data dictionary.
        If they are not, raises a KeyError.

        Args:
            data (Dict[str, Any]): The dataset containing 'uid' and 'text'.

        Returns:
            Dict[str, Any]: Results after running the prediction chain.

        Raises:
            KeyError: If either 'uid' or 'text' is not in data.
        """
        # Check if 'uid' and 'text' keys exist in data
        if 'uid' not in data.keys() or 'text' not in data.keys():
            raise KeyError("Both 'uid' and 'text' keys must be present in the data dictionary.")

        # Initialize the RateLimitedTaskExecutor with a rate limit of 20 requests per second
        task_executor = RateLimitedTaskExecutor(rate_limit=self.rate_limit, worker_func=self.run_text_bison_context)
        # Prepare the argument list for labeling
        args_list = [(uid, text) for uid, text in zip(data['uid'], data['text'])]
        # Execute the labeling tasks
        results = task_executor.execute(args_list)

        return results