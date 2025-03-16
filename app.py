import os
import json
import time
import re
import random
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO
from celery import Celery
import together
from openai import OpenAI
from anthropic import Anthropic
import groq
from datetime import timedelta
import traceback

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Use threading mode which is more compatible
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", logger=True)

app.config['broker_url'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(app.config)
celery.conf.result_expires = 3600

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("saved_conversations", exist_ok=True)

# Globals
seed = random.randint(1, 1000000)

def log_function_call(func):
    """Decorator that logs calls and results."""
    def wrapper(*args, **kwargs):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Don't log full prompts - we'll log processed prompts separately
        if func.__name__ in ["get_together_completion", "get_openai_completion", "get_claude_completion", "get_groq_completion"]:
            log_line = f"{timestamp} - {func.__name__} called with kwargs: {kwargs}\n"
        else:
            log_line = f"{timestamp} - {func.__name__} called with args truncated, kwargs: {kwargs}\n"

        with open("logs/function_calls.txt", "a", encoding="utf-8") as f:
            f.write(log_line)

        result = func(*args, **kwargs)

        # Truncate result in logs to avoid massive log files
        if isinstance(result, str) and len(result) > 200:
            result_summary = result[:200] + "..."
        else:
            result_summary = str(result)[:200] + "..." if len(str(result)) > 200 else result
            
        log_line_result = f"{timestamp} - {func.__name__} returned: {result_summary}\n\n"
        with open("logs/function_calls.txt", "a", encoding="utf-8") as f:
            f.write(log_line_result)

        return result
    return wrapper

def log_error(error_message):
    """Log errors to error log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("logs/error_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ERROR: {error_message}\n\n")
    socketio.emit('log_message', {'type': 'error', 'message': error_message})

def log_debug(debug_message):
    """Log debug information to debug log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("logs/debug_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] DEBUG: {debug_message}\n\n")
    socketio.emit('log_message', {'type': 'debug', 'message': debug_message})

def log_processed_prompt(prompt_name, processed_prompt):
    """Log the processed prompt after placeholder replacement."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a truncated version for logging (first 300 chars)
    truncated_prompt = processed_prompt[:300] + "..." if len(processed_prompt) > 300 else processed_prompt
    
    with open("logs/processed_prompts.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] PROCESSED {prompt_name}:\n{truncated_prompt}\n\n")
    
    socketio.emit('processed_prompt', {'name': prompt_name, 'content': truncated_prompt})

@log_function_call
def read_prompt_template(file_path):
    """Read a prompt template from file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        error_msg = f"Error reading prompt template {file_path}: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def extract_thinking(response):
    """
    Extract <think>...</think> tags from response.
    Returns:
      - The thinking content inside the tags
      - The response after the thinking tags
    """
    if not response or not isinstance(response, str):
        return "", response
        
    think_pattern = r'<think>(.*?)</think>\s*'
    think_match = re.search(think_pattern, response, re.DOTALL)
    
    if think_match:
        thinking = f"<think>{think_match.group(1)}</think>"
        response_after_thinking = re.sub(think_pattern, '', response, count=1, flags=re.DOTALL).strip()
        return thinking, response_after_thinking
    else:
        return "", response

def extract_function_calls(response):
    """
    Extract <function> search_func(...) </function> calls with improved error handling.
    Returns:
      - 'cleaned' user-facing response (with those calls removed)
      - a list of the function call body strings
    """
    if not response or not isinstance(response, str):
        log_error(f"Invalid response passed to extract_function_calls: {type(response)}")
        return "", []
        
    try:
        # Pattern to match various formats of function calls - try in order and stop after first match
        patterns = [
            r'<function>\s*search_func\((.*?)\)\s*</function>',  # Standard format
            r'<function>search_func\((.*?)\)</function>',        # No spaces
            r'<function>\s*search_func\s*\((.*?)\)\s*</function>', # Extra spaces
            r'<function>(.*?)</function>'  # More lenient - any content in function tags
        ]
        
        function_calls = []
        clean_response = response
        
        # Try each pattern until we find a match
        for pattern in patterns:
            calls = re.findall(pattern, response, flags=re.DOTALL)
            if calls:
                function_calls = calls  # Use only the matches from this pattern
                clean_response = re.sub(pattern, '', response, flags=re.DOTALL)
                log_debug(f"Extract function calls - Found pattern match: {pattern}")
                break  # Stop after finding first matching pattern
        
        log_debug(f"Extract function calls - Input length: {len(response)}")
        log_debug(f"Extract function calls - Found {len(function_calls)} function calls")
        
        clean_response = clean_response.strip()
        
        # Sanity check - if we removed everything, something might be wrong
        if not clean_response and function_calls:
            log_debug("Warning: After removing function calls, response is empty")
        
        return clean_response, function_calls
    
    except Exception as e:
        log_error(f"Unexpected error in extract_function_calls: {str(e)}")
        return response, []  # Return original response in case of error


def create_detailed_message(thinking, response_after_thinking, final_response, search_history=None, critique=None):
    """Create detailed assistant message with optimized structure"""
    msg = {
        "role": "assistant",
        "thinking": thinking
    }
    
    # Only add raw_response_after_thinking if it's different from content
    if response_after_thinking and response_after_thinking != final_response:
        msg["raw_response_after_thinking"] = response_after_thinking
        
    # Add search results if any
    if search_history:
        msg["search_history"] = search_history
    
    # Add critique if any
    if critique:
        msg["critique"] = critique
        
    # Add content (always required)
    msg["content"] = final_response
    
    return msg

def save_search_results(search_params, search_result, conversation_id):
    """Append search info to search_history.json and return the search record"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        search_record = {
            "timestamp": timestamp,
            "parameters": search_params,
            "results": search_result
        }
        
        # Emit to socket for real-time update
        socketio.emit('search_results', {
            'conversation_id': conversation_id,
            'record': search_record
        })
        
        try:
            with open('search_history.json', 'r', encoding='utf-8') as f:
                search_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            search_history = []
            
        search_history.append(search_record)
        with open('search_history.json', 'w', encoding='utf-8') as f:
            json.dump(search_history, f, ensure_ascii=False, indent=2)
            
        return search_record
    except Exception as e:
        error_msg = f"Error saving search results: {str(e)}"
        print(f"\n{error_msg}")
        log_error(error_msg)
        return {"error": str(e)}

def get_openai_client():
    """Initialize OpenAI client with API key."""
    try:
        with open("openai.key", "r", encoding="utf-8") as f:
            openai_key = f.read().strip()
        return OpenAI(api_key=openai_key)
    except Exception as e:
        error_msg = f"Error initializing OpenAI client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_together_client():
    """Initialize Together client with API key."""
    try:
        with open("together.key", "r", encoding="utf-8") as f:
            together_key = f.read().strip()
        return together.Together(api_key=together_key)
    except Exception as e:
        error_msg = f"Error initializing Together client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_claude_client():
    """Initialize Claude client with API key."""
    try:
        with open("claude.key", "r", encoding="utf-8") as f:
            claude_key = f.read().strip()
        return Anthropic(api_key=claude_key)
    except Exception as e:
        error_msg = f"Error initializing Claude client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

def get_groq_client():
    """Initialize Groq client with API key."""
    try:
        with open("groq.key", "r", encoding="utf-8") as f:
            groq_key = f.read().strip()
        return groq.Client(api_key=groq_key)
    except Exception as e:
        error_msg = f"Error initializing Groq client: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@celery.task(bind=True)
def get_openai_completion_task(self, prompt, model="o3-mini", conversation_id=None):
    """Celery task for OpenAI completion with file-based storage"""
    result = get_openai_completion(prompt, model)
    
    if conversation_id:
        # Emit the result via socketio
        socketio.emit('model_response', {
            'model': f'openai_{model}',
            'conversation_id': conversation_id,
            'response': result
        })
        
        # For persona generation (step 1), store result in both session and file
        if 'persona.txt' in prompt and result:
            # Store in session
            try:
                session_data = session.get('data', {})
                session_data['persona'] = result
                session_data['conversation_id'] = conversation_id
                session['data'] = session_data
            except Exception as e:
                log_error(f"Error storing persona in session: {str(e)}")
            
            # Also store in file for reliable retrieval
            save_conversation_data(conversation_id, "persona", result)
            log_debug(f"Stored persona in file for conversation {conversation_id}")
        
        # For requirements, store both in session and file
        if 'requirement.txt' in prompt and result:
            # Store in session
            try:
                session_data = session.get('data', {})
                session_data['requirements'] = result
                session['data'] = session_data
            except Exception as e:
                log_error(f"Error storing requirements in session: {str(e)}")
            
            # Also store in file
            save_conversation_data(conversation_id, "requirements", result)
            log_debug(f"Stored requirements in file for conversation {conversation_id}")
        
        # Log event emission
        log_debug(f"Emitted model_response event for {model} with conversation_id {conversation_id}")
    
    return result

@log_function_call
def get_openai_completion(prompt, model="o3-mini"):
    """
    Use OpenAI for responses.
    Returns the response as a string.
    """
    try:
        # Log the processed prompt
        log_processed_prompt(f"OpenAI_{model}", prompt)
        
        client = get_openai_client()
        if not client:
            return None
            
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            seed=seed
        )
        return completion.choices[0].message.content if completion.choices else None

    except Exception as e:
        error_msg = f"Error in get_openai_completion for model {model}: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@celery.task(bind=True)
def get_together_completion_task(self, prompt, include_thinking=False, conversation_id=None):
    """Celery task for Together AI completion"""
    result = get_together_completion(prompt, include_thinking)
    if conversation_id:
        socketio.emit('model_response', {
            'model': 'together',
            'conversation_id': conversation_id,
            'response': result
        })
    return result

@log_function_call
def get_together_completion(prompt, include_thinking=False):
    """
    Use Together AI DeepSeek-R1 for responses.
    Returns the response as a string.
    """
    try:
        # Log the processed prompt
        log_processed_prompt("Together_DeepSeek-R1", prompt)
        
        client = get_together_client()
        if not client:
            return None
            
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            stream=False,
            seed=seed
        )
        final_text = completion.choices[0].message.content if completion.choices else ""
        
        # Return the full text including thinking if requested
        if include_thinking:
            return final_text if final_text.strip() else None
        else:
            # Otherwise remove thinking tags
            cleaned_text = re.sub(r'<think>.*?</think>\s*', '', final_text, flags=re.DOTALL)
            return cleaned_text if cleaned_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_together_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@celery.task(bind=True)
def get_claude_completion_task(self, prompt, include_thinking=False, conversation_id=None):
    """Celery task for Claude completion"""
    result = get_claude_completion(prompt, include_thinking)
    if conversation_id:
        socketio.emit('model_response', {
            'model': 'claude',
            'conversation_id': conversation_id,
            'response': result
        })
    return result

@log_function_call
def get_claude_completion(prompt, include_thinking=False):
    """
    Use Claude for responses.
    Returns the response as a string.
    """
    try:
        # Log the processed prompt
        log_processed_prompt("Claude_3-7-sonnet", prompt)
        
        client = get_claude_client()
        if not client:
            return None

        completion = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        final_text = completion.content[0].text if completion.content else ""
        
        return final_text if final_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_claude_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@celery.task(bind=True)
def get_groq_completion_task(self, prompt, include_thinking=False, conversation_id=None):
    """Celery task for Groq completion"""
    result = get_groq_completion(prompt, include_thinking)
    if conversation_id:
        socketio.emit('model_response', {
            'model': 'groq',
            'conversation_id': conversation_id,
            'response': result
        })
    return result

@log_function_call
def get_groq_completion(prompt, include_thinking=False):
    """
    Use Groq DeepSeek-R1-Distill-Qwen-32B for responses.
    Returns the response as a string.
    """
    try:
        # Log the processed prompt
        log_processed_prompt("Groq_DeepSeek", prompt)
        
        client = get_groq_client()
        if not client:
            return None

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            stream=False,
            seed=seed
        )
        final_text = completion.choices[0].message.content if completion.choices else ""

        # Return the full text including thinking if requested
        if include_thinking:
            return final_text if final_text.strip() else None
        else:
            # Otherwise remove thinking tags
            cleaned_text = re.sub(r'<think>.*?</think>\s*', '', final_text, flags=re.DOTALL)
            return cleaned_text if cleaned_text.strip() else None

    except Exception as e:
        error_msg = f"Error in get_groq_completion: {str(e)}"
        print("\n" + error_msg)
        log_error(error_msg)
        return None

@celery.task(bind=True)
def extract_ner_from_conversation_task(self, conversation_history, conversation_id):
    """Celery task for NER extraction with improved error handling"""
    try:
        result = extract_ner_from_conversation(conversation_history)
        socketio.emit('ner_extracted', {
            'conversation_id': conversation_id,
            'preferences': result
        })
        # Save NER results to file for reliable retrieval
        save_conversation_data(conversation_id, "ner_results", result)
        return result
    except Exception as e:
        error_msg = f"Error in NER extraction task: {str(e)}\n{traceback.format_exc()}"
        log_error(error_msg)
        socketio.emit('log_message', {'type': 'error', 'message': error_msg})
        return {"error": str(e)}

def extract_ner_from_conversation(conversation_history):
    """
    Extract named entities (hotel preferences) from conversation using NER prompt.
    Improved to handle different conversation formats.
    """
    try:
        # Create simple conversation history without thinking for passing to LLMs
        simple_conversation = []
        
        if not conversation_history:
            log_error("extract_ner_from_conversation: Empty conversation history")
            return {}
            
        # Handle different conversation formats
        for msg in conversation_history:
            if isinstance(msg, dict):
                if "role" in msg and "content" in msg:
                    # Standard format
                    simple_conversation.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                elif "role" in msg and isinstance(msg.get("content"), dict):
                    # Handle nested content structure
                    content = msg.get("content", {})
                    if isinstance(content, dict) and "text" in content:
                        simple_conversation.append({
                            "role": msg["role"],
                            "content": content["text"]
                        })
            elif isinstance(msg, str):
                # String format - assume user message
                simple_conversation.append({
                    "role": "user",
                    "content": msg
                })
                
        if not simple_conversation:
            log_error("extract_ner_from_conversation: Failed to parse conversation")
            # Log the conversation structure to help debug
            log_debug(f"Conversation structure: {type(conversation_history)}")
            if isinstance(conversation_history, list) and len(conversation_history) > 0:
                log_debug(f"First message type: {type(conversation_history[0])}")
            return {}
        
        # Add debug - log conversation length
        log_debug(f"NER extraction - processing conversation with {len(simple_conversation)} messages")
        
        # Read NER prompt template
        ner_template = read_prompt_template("ner.md")
        if not ner_template:
            log_error("Failed to read ner.md template")
            return {}
            
        # Prepare NER prompt
        ner_prompt = ner_template.replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        
        # Get NER extraction results using OpenAI
        ner_response = get_openai_completion(ner_prompt, model="o3-mini")
        
        if not ner_response:
            log_error("No NER response generated")
            return {}
            
        # Add debug - print part of response    
        log_debug(f"NER extraction response preview: {ner_response[:100]}...")
            
        # Extract Python dictionary from response
        dict_pattern = r'```python\s*({[\s\S]*?})\s*```'
        dict_match = re.search(dict_pattern, ner_response)
        
        if dict_match:
            try:
                # Parse the extracted dictionary string
                dict_str = dict_match.group(1)
                # Replace any None with "None" for safer eval
                dict_str = dict_str.replace(": None", ': "None"')
                preferences_dict = eval(dict_str)
                log_debug(f"Extracted preferences: {json.dumps(preferences_dict, ensure_ascii=False)}")
                return preferences_dict
            except Exception as e:
                log_error(f"Error parsing extracted preferences: {str(e)}")
                # Try a more lenient approach on failure
                try:
                    # Remove any problematic characters and try again
                    dict_str = dict_match.group(1)
                    dict_str = re.sub(r',\s*}', '}', dict_str)  # Remove trailing commas
                    preferences_dict = eval(dict_str)
                    log_debug(f"Extracted preferences (second attempt): {json.dumps(preferences_dict, ensure_ascii=False)}")
                    return preferences_dict
                except Exception as e2:
                    log_error(f"Error in second attempt parsing preferences: {str(e2)}")
                    return {}
        else:
            # Try direct extraction if no code block is found
            try:
                # Look for a dictionary pattern without code block
                dict_pattern = r'({[\s\S]*?})'
                dict_match = re.search(dict_pattern, ner_response)
                if dict_match:
                    dict_str = dict_match.group(1)
                    # Replace any None with "None" for safer eval
                    dict_str = dict_str.replace(": None", ': "None"')
                    preferences_dict = eval(dict_str)
                    log_debug(f"Extracted preferences (direct): {json.dumps(preferences_dict, ensure_ascii=False)}")
                    return preferences_dict
                else:
                    log_error("No valid preferences dictionary found in NER response")
                    return {}
            except Exception as e:
                log_error(f"Error with direct parsing: {str(e)}")
                return {}
    except Exception as e:
        error_msg = f"Error in NER extraction: {str(e)}\n{traceback.format_exc()}"
        log_error(error_msg)
        return {}

@celery.task(bind=True)
def process_search_call_task(self, extracted_preferences, conversation_id):
    """Celery task for search call processing with improved error handling"""
    try:
        result = process_search_call(extracted_preferences)
        socketio.emit('search_call_processed', {
            'conversation_id': conversation_id,
            'search_call': result
        })
        # Save search call to file for reliable retrieval
        save_conversation_data(conversation_id, "search_call", result)
        return result
    except Exception as e:
        error_msg = f"Error in search call task: {str(e)}\n{traceback.format_exc()}"
        log_error(error_msg)
        socketio.emit('log_message', {'type': 'error', 'message': error_msg})
        return ""

def process_search_call(extracted_preferences):
    """
    Determine if a search should be triggered based on extracted preferences.
    Improved to handle different preference formats and better error reporting.
    """
    try:
        # Add debug for empty preferences check
        if not extracted_preferences:
            log_debug("Process search call - preferences empty, still checking search need")
            
        # Ensure preferences is a valid dictionary
        if not isinstance(extracted_preferences, dict):
            log_error(f"Invalid preferences format: {type(extracted_preferences)}")
            extracted_preferences = {}
            
        # Read search call prompt template
        search_call_template = read_prompt_template("search_call.md")
        if not search_call_template:
            log_error("Failed to read search_call.md template")
            return ""
            
        # Prepare search call prompt with safer JSON handling
        try:
            preferences_json = json.dumps(extracted_preferences, ensure_ascii=False, indent=2)
        except Exception as e:
            log_error(f"Error converting preferences to JSON: {str(e)}")
            preferences_json = "{}"
            
        search_call_prompt = search_call_template.replace("{preferences}", preferences_json)
        
        # Get search call decision using OpenAI
        search_call_response = get_openai_completion(search_call_prompt, model="o3-mini")
        
        if not search_call_response:
            log_error("No search call response generated")
            return ""
            
        # Add debug - print response preview
        log_debug(f"Search call response preview: {search_call_response[:100]}...")
            
        # Clean and return the response
        search_call_response = search_call_response.strip()
        
        # Check if the response is NO_SEARCH_NEEDED
        if "NO_SEARCH_NEEDED" in search_call_response:
            log_debug("Search call explicitly indicates no search needed")
            return ""
            
        # Only return if it's a search function call
        if "<function>" in search_call_response:
            log_debug("Search function call detected in search_call response")
            return search_call_response
        
        # Otherwise return empty string
        log_debug("No function call found in search call response")
        return ""
    except Exception as e:
        error_msg = f"Error in search call processing: {str(e)}\n{traceback.format_exc()}"
        log_error(error_msg)
        return ""

@celery.task(bind=True)
def process_search_simulation_task(self, response_after_thinking, conversation_history, conversation_id):
    """Celery task for search simulation"""
    result = process_search_simulation_openai(response_after_thinking, conversation_history, conversation_id)
    return result

def process_search_simulation_openai(response_after_thinking, conversation_history, conversation_id=None):
    """Process search simulation with OpenAI and return search record for logging"""
    if not response_after_thinking or response_after_thinking.strip() == "":
        log_error("Empty response passed to search processing")
        return None
    
    log_debug(f"Processing search in response of length: {len(response_after_thinking)}")
    
    try:
        clean_response, function_calls = extract_function_calls(response_after_thinking)
        
        if not function_calls:
            log_debug("No function calls detected - returning None")
            return None
        
        log_debug(f"Found {len(function_calls)} function calls to process")
        
        # We'll process only the first function call (if multiple)
        function_call_content = function_calls[0]
        
        log_debug(f"Processing search query: {function_call_content}")
        search_template = read_prompt_template("search_simulator.md")
        if not search_template:
            log_error("Failed to read search_simulator.md template")
            return None
            
        search_prompt = search_template.replace("{search_query}", function_call_content.strip())
        search_response = get_openai_completion(search_prompt)
        
        if not search_response:
            log_error("No search result received for query")
            return None
            
        log_debug(f"Search result received of length: {len(search_response)}")
        
        # Save and return the search record
        search_record = save_search_results(function_call_content, search_response, conversation_id)
        return search_record
        
    except Exception as e:
        error_msg = f"Unexpected error in search processing: {str(e)}"
        log_error(error_msg)
        return None

@celery.task(bind=True)
def get_critic_evaluation_task(self, original_prompt, conversation_history, search_record, assistant_response, conversation_id):
    """Celery task for critic evaluation"""
    result = get_critic_evaluation_together(original_prompt, conversation_history, search_record, assistant_response)
    socketio.emit('critic_evaluation', {
        'conversation_id': conversation_id,
        'critique': result
    })
    return result

def get_critic_evaluation_together(original_prompt, conversation_history, search_record, assistant_response):
    """
    Get a critique of the assistant's response using the critic.md prompt with DeepSeek-R1.
    Returns a JSON object with score, reason, and thinking.
    """
    try:
        # Create simple conversation history without thinking for passing to LLMs
        simple_conversation = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in conversation_history
        ]
        
        # Read critic prompt template
        critic_template = read_prompt_template("critic.md")
        if not critic_template:
            log_error("Failed to read critic.md template")
            return None
            
        # Prepare placeholder replacements
        critic_prompt = critic_template.replace("{original_prompt}", original_prompt)
        critic_prompt = critic_prompt.replace("{conversation}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        critic_prompt = critic_prompt.replace("{last_response}", assistant_response)
        
        # Only include search history in critic input if it was also shown to the assistant
        if search_record and "show_results_to_actor" in search_record and search_record["show_results_to_actor"] == True:
            critic_prompt = critic_prompt.replace("{search_history}", json.dumps(search_record["results"], ensure_ascii=False, indent=2))
        else:
            # If no search results were shown to assistant, don't include them for the critic either
            critic_prompt = critic_prompt.replace("<last_search_output>\n{search_history}\n</last_search_output>", "")
        
        # Get critique using Together API with include_thinking=True to get the thinking tags
        critique_response_with_thinking = get_together_completion(critic_prompt, include_thinking=True)
        
        if not critique_response_with_thinking:
            log_error("No critique response generated from Together API")
            return None
        
        # Extract thinking tags from the response
        thinking, critique_response = extract_thinking(critique_response_with_thinking)
        
        # Try to parse the JSON response
        try:
            # Clean up any extra text that might be around the JSON
            json_pattern = r'(\{[\s\S]*\})'
            json_match = re.search(json_pattern, critique_response)
            
            if json_match:
                critique_json = json.loads(json_match.group(1))
                # Add the thinking part to the JSON
                critique_json["thinking"] = thinking
                log_debug(f"Parsed Together critique: {json.dumps(critique_json, ensure_ascii=False)}")
                return critique_json
            else:
                log_error("No valid JSON found in Together critique response")
                return None
        except Exception as e:
            log_error(f"Error parsing Together critique JSON: {str(e)}")
            return None
            
    except Exception as e:
        error_msg = f"Error in Together critique evaluation: {str(e)}"
        log_error(error_msg)
        return None

@celery.task(bind=True)
def process_search_results_task(self, search_record, conversation_id):
    """Celery task for processing search results"""
    show_results, search_text = process_search_results(search_record)
    socketio.emit('search_results_processed', {
        'conversation_id': conversation_id,
        'show_results': show_results,
        'search_text': search_text
    })
    return show_results, search_text

@celery.task(bind=True)
def get_combined_critique_task(self, original_prompt, conversation_history, search_record, assistant_response, conversation_id):
    """Celery task for combined critique"""
    result = get_combined_critique(original_prompt, conversation_history, search_record, assistant_response)
    socketio.emit('combined_critique', {
        'conversation_id': conversation_id,
        'critique': result
    })
    return result

@celery.task(bind=True)
def process_model_response_task(self, model_type, conversation, enable_search, evaluate_responses, previous_data, conversation_id):
    """Celery task for processing model response"""
    result = process_model_response(model_type, conversation, enable_search, evaluate_responses, previous_data, conversation_id)
    socketio.emit('model_response_processed', {
        'conversation_id': conversation_id,
        'result': result
    })
    return result

# Add this fixed version of the process_model_response function

def process_model_response(model_type, conversation, ENABLE_SEARCH, EVALUATE_RESPONSES, previous_data=None, conversation_id=None):
    """Process a single response from a specific model with optimized search"""
    log_debug(f"Processing {model_type.upper()} response...")
    
    # Initialize previous data if None
    if previous_data is None:
        previous_data = {
            "preferences": {},
            "num_matches": 100  # Default high number to ensure first search happens
        }
    
    # Create simple conversation history without thinking for passing to LLMs
    simple_conversation = []
    for msg in conversation:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            simple_conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        else:
            log_debug(f"Skipping invalid message in conversation: {msg}")
    
    # Log the simplified conversation
    log_debug(f"Simple conversation has {len(simple_conversation)} messages")
    
    # --- STEP 1: Extract NER from conversation ---
    extracted_preferences = {}
    search_record = None
    show_results_to_actor = False
    num_matches = previous_data.get("num_matches", 100)  # Use get() with default
    
    if ENABLE_SEARCH:
        # Always extract preferences to check if they changed
        extracted_preferences = extract_ner_from_conversation(simple_conversation)
        log_debug(f"{model_type}: Extracted preferences: {json.dumps(extracted_preferences, ensure_ascii=False)}")
    
        # Calculate key conditions for search triggering
        preferences_changed = extracted_preferences != previous_data.get("preferences", {})  # Use get() with default
        many_previous_results = previous_data.get("num_matches", 100) > 10  # Use get() with default
        
        log_debug(f"{model_type}: Preferences changed: {preferences_changed}")
        log_debug(f"{model_type}: Previous search had many results (> 10): {many_previous_results}")
        log_debug(f"{model_type}: Previous search matches: {previous_data.get('num_matches', 100)}")
        log_debug(f"{model_type}: Previous preferences: {json.dumps(previous_data.get('preferences', {}), ensure_ascii=False)}")
        log_debug(f"{model_type}: New preferences: {json.dumps(extracted_preferences, ensure_ascii=False)}")
    
        # --- STEP 2: Determine if search should be triggered ---
        # RULE 1: Always call search if previous results > 10, regardless of preference change
        # RULE 2: When previous results <= 10, only call search if preferences changed
        should_trigger_search = many_previous_results or (preferences_changed and not many_previous_results)
        
        if should_trigger_search:
            if many_previous_results:
                log_debug(f"{model_type}: Search triggered because previous results > 10")
            else:
                log_debug(f"{model_type}: Search triggered because preferences changed and previous results <= 10")
                
            search_call_result = process_search_call(extracted_preferences)
            log_debug(f"{model_type}: Search call result length: {len(search_call_result) if search_call_result else 0}")
        
            # --- STEP 3: Process search if needed ---
            if search_call_result:
                log_debug(f"{model_type}: Processing search results")
                try:
                    search_record = process_search_simulation_openai(
                        search_call_result,
                        simple_conversation,
                        conversation_id
                    )
                    log_debug(f"{model_type}: Search record created and logged: {search_record is not None}")
                    
                    # Extract number of matches immediately after search
                    if search_record:
                        search_response = search_record.get("results", "")
                        if search_response:
                            # Try to extract the number of matches using various patterns
                            patterns = [
                                r'"Number of matches":\s*(\d+)',
                                r'Number of matches:\s*(\d+)',
                                r'Found (\d+) matches',
                                r'(\d+) results found',
                                r'(\d+) hotels match'
                            ]
                            
                            for pattern in patterns:
                                matches_match = re.search(pattern, search_response, re.IGNORECASE)
                                if matches_match:
                                    try:
                                        num_matches = int(matches_match.group(1))
                                        log_debug(f"Found {num_matches} matches in search results using pattern: {pattern}")
                                        break
                                    except ValueError:
                                        continue
                            
                            # Fall back to other detection methods if needed
                            if num_matches == 0:
                                # Count hotel entities as a fallback
                                hotel_count = len(re.findall(r'Hotel\d+', search_response, re.IGNORECASE))
                                if hotel_count > 0:
                                    num_matches = hotel_count
                                    log_debug(f"Fallback: Estimated {num_matches} matches by counting Hotel entities")
                        
                        # Store the number of matches in the search record
                        search_record["num_matches"] = num_matches
                        log_debug(f"{model_type}: Extracted {num_matches} matches from search results")
                
                except Exception as e:
                    error_msg = f"{model_type}: Error during search processing: {str(e)}"
                    log_error(error_msg)
                    print(f"WARNING: {error_msg}")
        else:
            if preferences_changed:
                log_debug(f"{model_type}: No search triggered: preferences changed but previous results <= 10")
            else:
                log_debug(f"{model_type}: No search triggered: preferences unchanged and previous results <= 10")
    
    # --- STEP 4: Process search results to determine if they should be shown to actor ---
    search_text = ""
    
    if search_record:
        # Show search results to actor when number of matches < 50
        if search_record.get("num_matches", 100) < 50:
            show_results_to_actor = True
            search_text = search_record.get("results", "")
            log_debug(f"{model_type}: Search results will be shown to actor ({search_record.get('num_matches', 100)} matches < 50)")
        else:
            show_results_to_actor = False
            log_debug(f"{model_type}: Search results will NOT be shown to actor ({search_record.get('num_matches', 100)} matches >= 50)")
        
        # Add the show_results_to_actor flag to the search record for reference by critic
        search_record["show_results_to_actor"] = show_results_to_actor
        
        # Update num_matches from search_record for consistency
        num_matches = search_record.get("num_matches", 100)
    else:
        # If we didn't run a search this time, use the previous number of matches
        num_matches = previous_data.get("num_matches", 100)
        log_debug(f"{model_type}: Using previous num_matches: {num_matches} (no new search)")
    
    # --- STEP 5: Generate assistant response ---
    # Read actor prompt template
    agent_template = read_prompt_template("actor.md")
    if not agent_template:
        log_error(f"{model_type}: Failed to read actor.md template")
        return None
    
    # Prepare actor prompt - include num_matches only if search results exist
    agent_prompt = (
        agent_template
        .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        .replace("{search}", search_text if show_results_to_actor else "")
        .replace("{num_matches}", str(num_matches) if show_results_to_actor else "")
    )
    
    # Get assistant response based on model type
    assistant_response = None
    if model_type == 'claude':
        assistant_response = get_claude_completion(agent_prompt)
    elif model_type == 'together':
        assistant_response = get_together_completion(agent_prompt)
    elif model_type == 'groq':
        assistant_response = get_groq_completion(agent_prompt)
    
    if not assistant_response:
        log_debug(f"No {model_type} assistant response generated; skipping.")
        print(f"No {model_type} assistant response generated; skipping.")
        return None
    
    # Extract thinking (if any) from the response
    thinking, response_after_thinking = extract_thinking(assistant_response)
    
    # Check if the final response still has function calls and clean them
    final_response, _ = extract_function_calls(response_after_thinking)
    if not final_response or final_response.strip() == "":
        final_response = response_after_thinking
    
    # --- STEP 6: Evaluate the response with critics if enabled ---
    critique = None
    if EVALUATE_RESPONSES:
        original_prompt = read_prompt_template("actor.md")
        if original_prompt:
            original_prompt = original_prompt.replace("{conv}", "").replace("{search}", "").replace("{num_matches}", "").strip()
            
            try:
                critique = get_combined_critique(
                    original_prompt,
                    simple_conversation,
                    search_record,
                    final_response
                )
                
                if critique:
                    together_score = critique.get("together", {}).get("score", "N/A") if critique.get("together") else "N/A"
                    log_debug(f"{model_type}: Together critique score: {together_score}")
            except Exception as e:
                log_error(f"Error in critique evaluation: {str(e)}")
                critique = None
    
    assistant_msg = create_detailed_message(
        thinking,
        response_after_thinking,
        final_response,
        search_record,
        critique
    )
    
    conversation.append(assistant_msg)
    
    log_debug(f"Assistant ({model_type}): {final_response[:100]}...")
    
    return {
        "message": assistant_msg,
        "data": {
            "preferences": extracted_preferences,
            "num_matches": num_matches
        }
    }

def debug_search_status(model_type, ENABLE_SEARCH, previous_data):
    """
    Debug function to check and log search status
    """
    if not ENABLE_SEARCH:
        log_debug(f"{model_type}: SEARCH DISABLED globally")
        return
    
    if previous_data is None:
        log_debug(f"{model_type}: No previous data exists - search will happen on first run")
        return
    
    num_matches = previous_data.get("num_matches", 100)
    log_debug(f"{model_type}: Current num_matches: {num_matches}")
    
    preferences = previous_data.get("preferences", {})
    
    if num_matches <= 10:
        log_debug(f"{model_type}: Search MAY TRIGGER on next turn if preferences change (num_matches <= 10)")
    else:
        log_debug(f"{model_type}: SEARCH WILL NOT TRIGGER on next turn (num_matches > 10)")
        
    log_debug(f"{model_type}: Current preferences: {json.dumps(preferences, ensure_ascii=False)}")

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init_conversation():
    """Initialize a new conversation"""
    conversation_id = str(uuid.uuid4())
    random_number = random.randint(1, 100)
    session_data = {
        'conversation_id': conversation_id,
        'random_number': random_number,
        'conversation': [],
        'persona': None,
        'requirements': None,
        'previous_data': {
            'preferences': {},
            'num_matches': 100
        },
        'saved_conversations': []
    }
    session['data'] = session_data
    
    persona_template = read_prompt_template("persona.txt")
    persona_template = re.sub(r'\{number\}', str(random_number), persona_template)
    
    task = get_openai_completion_task.delay(persona_template, model="o3-mini", conversation_id=conversation_id)
    
    return jsonify({
        'status': 'success',
        'conversation_id': conversation_id,
        'task_id': task.id
    })

@app.route('/generate_requirements', methods=['POST'])
def generate_requirements():
    """Generate requirements based on persona"""
    data = request.get_json()
    persona = data.get('persona')
    conversation_id = data.get('conversation_id')
    
    session_data = session.get('data', {})
    session_data['persona'] = persona
    session['data'] = session_data
    
    requirements_template = read_prompt_template("requirement.txt")
    requirements_prompt = requirements_template.replace("{persona}", persona)
    
    task = get_openai_completion_task.delay(
        requirements_prompt, 
        model="o3-mini",
        conversation_id=conversation_id
    )
    
    return jsonify({
        'status': 'success',
        'task_id': task.id
    })

@app.route('/initial_user_message', methods=['POST'])
def initial_user_message():
    """Generate the initial user message"""
    data = request.get_json()
    requirements = data.get('requirements')
    conversation_id = data.get('conversation_id')
    
    session_data = session.get('data', {})
    session_data['requirements'] = requirements
    session['data'] = session_data
    
    user_sim_template = read_prompt_template("user_simulator.txt")
    initial_user_prompt = (
        user_sim_template
        .replace("{conv}", "[]")
        .replace("{requirements}", requirements)
        .replace("{persona}", session_data['persona'])
    )
    
    task = get_openai_completion_task.delay(
        initial_user_prompt, 
        model="o3-mini",
        conversation_id=conversation_id
    )
    
    return jsonify({
        'status': 'success',
        'task_id': task.id
    })

@app.route('/add_user_message', methods=['POST'])
def add_user_message():
    """Add user message to conversation and process"""
    data = request.get_json()
    message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    # Update session with new user message
    try:
        session_data = session.get('data', {})
    except Exception as e:
        # If session can't be retrieved, try to get from file
        log_error(f"Error getting session data: {str(e)}")
        session_data = {
            'conversation_id': conversation_id,
            'conversation': [],
            'previous_data': {
                'preferences': {},
                'num_matches': 100
            },
            'saved_conversations': []
        }
    
    conversation = session_data.get('conversation', [])
    
    # Add user message
    user_msg = {"role": "user", "content": message}
    conversation.append(user_msg)
    
    # Update session
    session_data['conversation'] = conversation
    try:
        session['data'] = session_data
    except Exception as e:
        log_error(f"Error updating session: {str(e)}")
    
    # Also save to file for backup
    save_conversation_data(conversation_id, "conversation", conversation)
    
    return jsonify({
        'status': 'success',
        'message': message
    })

@app.route('/process_model_response', methods=['POST'])
def process_model_response_route():
    """Process model response"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    session_data = session.get('data', {})
    conversation = session_data.get('conversation', [])
    previous_data = session_data.get('previous_data', {
        'preferences': {},
        'num_matches': 100
    })
    
    task = process_model_response_task.delay(
        'claude',
        conversation,
        True,
        True,
        previous_data,
        conversation_id
    )
    
    return jsonify({
        'status': 'success',
        'task_id': task.id
    })

@app.route('/generate_next_user_message', methods=['POST'])
def generate_next_user_message():
    """Generate the next user message"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    session_data = session.get('data', {})
    conversation = session_data.get('conversation', [])
    requirements = session_data.get('requirements')
    persona = session_data.get('persona')
    
    simple_conversation = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in conversation
    ]
    
    user_sim_template = read_prompt_template("user_simulator.txt")
    user_prompt = (
        user_sim_template
        .replace("{conv}", json.dumps(simple_conversation, ensure_ascii=False, indent=2))
        .replace("{requirements}", requirements)
        .replace("{persona}", persona)
    )
    
    task = get_openai_completion_task.delay(
        user_prompt, 
        model="o3-mini",
        conversation_id=conversation_id
    )
    
    return jsonify({
        'status': 'success',
        'task_id': task.id
    })

@app.route('/debug_conversation', methods=['GET'])
def debug_conversation():
    """Debug endpoint to check conversation data"""
    conversation_id = request.args.get('conversation_id')
    
    # Get data from all possible sources
    response_data = {
        'session': None,
        'file': None,
        'memory': None
    }
    
    # Check session
    try:
        session_data = session.get('data', {})
        if session_data and 'conversation' in session_data:
            response_data['session'] = {
                'conversation_length': len(session_data['conversation']),
                'has_conversation': len(session_data['conversation']) > 0,
                'conversation_preview': [
                    {'role': msg.get('role'), 'content_preview': msg.get('content', '')[:50] + '...' if len(msg.get('content', '')) > 50 else msg.get('content', '')}
                    for msg in session_data['conversation'][:2]
                ] if session_data['conversation'] else []
            }
    except Exception as e:
        response_data['session_error'] = str(e)
    
    # Check file storage
    try:
        file_conversation = load_conversation_data(conversation_id, "conversation")
        if file_conversation:
            response_data['file'] = {
                'conversation_length': len(file_conversation),
                'has_conversation': len(file_conversation) > 0,
                'conversation_preview': [
                    {'role': msg.get('role'), 'content_preview': msg.get('content', '')[:50] + '...' if len(msg.get('content', '')) > 50 else msg.get('content', '')}
                    for msg in file_conversation[:2]
                ] if file_conversation else []
            }
    except Exception as e:
        response_data['file_error'] = str(e)
    
    return jsonify(response_data)



@app.route('/update_conversation', methods=['POST'])
def update_conversation():
    """Update the conversation with model response result"""
    data = request.get_json()
    result = data.get('result')
    conversation_id = data.get('conversation_id')
    
    if not result:
        return jsonify({
            'status': 'error',
            'message': 'No result provided'
        })
    
    # Update session
    try:
        session_data = session.get('data', {})
    except Exception as e:
        # If session can't be retrieved, try to get from file
        log_error(f"Error getting session data: {str(e)}")
        session_data = {
            'conversation_id': conversation_id,
            'conversation': [],
            'previous_data': {
                'preferences': {},
                'num_matches': 100
            },
            'saved_conversations': []
        }
    
    # Update previous data for next iteration
    if 'data' in result:
        session_data['previous_data'] = result['data']
    
    # Update conversation if message is present
    if 'message' in result:
        conversation = session_data.get('conversation', [])
        conversation.append(result['message'])
        session_data['conversation'] = conversation
        
        # Save conversation to file as backup
        save_conversation_data(conversation_id, "conversation", conversation)
    
    try:
        session['data'] = session_data
    except Exception as e:
        log_error(f"Error updating session: {str(e)}")
    
    return jsonify({
        'status': 'success',
        'conversation': session_data.get('conversation', [])
    })

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    """Save the current conversation"""
    data = request.get_json()
    name = data.get('name', f"Conversation-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    conversation_id = data.get('conversation_id')
    
    # Get session data
    session_data = session.get('data', {})
    conversation = session_data.get('conversation', [])
    
    if not conversation:
        # Try to get conversation from file if not in session
        conversation_file = os.path.join("conversation_data", conversation_id, "conversation.json")
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                    # Update session data
                    session_data['conversation'] = conversation
                    session['data'] = session_data
            except Exception as e:
                log_error(f"Error loading conversation from file: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to load conversation'
                })
    
    if not conversation:
        return jsonify({
            'status': 'error',
            'message': 'No conversation data available to save'
        })
    
    # Create readable version that doesn't include thinking
    readable_conversation = []
    for msg in conversation:
        if isinstance(msg, dict) and 'role' in msg:
            if msg['role'] == 'assistant':
                # For assistant messages, include critique scores if available
                critique_scores = {}
                if 'critique' in msg and msg['critique']:
                    if 'together' in msg['critique'] and msg['critique']['together']:
                        critique_scores['together'] = msg['critique']['together'].get('score', 'N/A')
                
                readable_msg = {
                    'role': 'assistant',
                    'content': msg['content']
                }
                
                if critique_scores:
                    readable_msg['critique_scores'] = critique_scores
                
                readable_conversation.append(readable_msg)
            else:
                # For user messages, just include role and content
                readable_conversation.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
    
    # Save conversation to file
    os.makedirs("saved_conversations", exist_ok=True)
    conv_filename = f"saved_conversations/{name}_readable.json"
    detailed_filename = f"saved_conversations/{name}_detailed.json"
    
    try:
        with open(conv_filename, 'w', encoding='utf-8') as f:
            json.dump(readable_conversation, f, ensure_ascii=False, indent=2)
        
        with open(detailed_filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        # Add debug log for successful save
        log_debug(f"Successfully saved conversation to {conv_filename} and {detailed_filename}")
        
        # Add to saved conversations list
        saved_conversation = {
            'name': name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'readable_file': conv_filename,
            'detailed_file': detailed_filename
        }
        
        session_data['saved_conversations'].append(saved_conversation)
        session['data'] = session_data
        
        # Also save to file for backup
        save_conversation_data(conversation_id, "saved_conversations", session_data['saved_conversations'])
        
        return jsonify({
            'status': 'success',
            'saved_conversation': saved_conversation
        })
    except Exception as e:
        error_msg = f"Error saving conversation: {str(e)}"
        log_error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg
        })


@app.route('/get_saved_conversations', methods=['GET'])
def get_saved_conversations():
    """Get list of saved conversations"""
    session_data = session.get('data', {})
    saved_conversations = session_data.get('saved_conversations', [])
    
    return jsonify({
        'status': 'success',
        'conversations': saved_conversations
    })

@app.route('/regenerate_assistant_response', methods=['POST'])
def regenerate_assistant_response():
    """Regenerate the last assistant response"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    session_data = session.get('data', {})
    conversation = session_data.get('conversation', [])
    
    if conversation and len(conversation) > 0:
        if conversation[-1].get('role') == 'assistant':
            conversation.pop()
    
    session_data['conversation'] = conversation
    session['data'] = session_data
    
    previous_data = session_data.get('previous_data', {
        'preferences': {},
        'num_matches': 100
    })
    
    task = process_model_response_task.delay(
        'claude',
        conversation,
        True,
        True,
        previous_data,
        conversation_id
    )
    
    return jsonify({
        'status': 'success',
        'task_id': task.id
    })

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear the current conversation but keep persona and requirements"""
    session_data = session.get('data', {})
    
    session_data['conversation'] = []
    session_data['previous_data'] = {
        'preferences': {},
        'num_matches': 100
    }
    
    session['data'] = session_data
    
    return jsonify({
        'status': 'success'
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset everything and start with a new conversation"""
    session.clear()
    
    return jsonify({
        'status': 'success'
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    log_debug('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    log_debug('Client disconnected')

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Get the status of a Celery task"""
    task = celery.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'status': 'Failed',
            'error': str(task.info)
        }
    else:
        response = {
            'state': task.state,
            'status': 'Success' if task.state == 'SUCCESS' else task.state
        }
        if task.info:
            if isinstance(task.info, dict) and 'result' in task.info:
                response['result'] = task.info['result']
            else:
                response['result'] = str(task.info)[:100] + '...' if len(str(task.info)) > 100 else str(task.info)
    
    return jsonify(response)

@app.route('/get_persona', methods=['GET'])
def get_persona():
    """Get the stored persona for a conversation"""
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    session_data = session.get('data', {})
    if session_data.get('conversation_id') == conversation_id and session_data.get('persona'):
        return jsonify({
            'status': 'success',
            'persona': session_data['persona'],
            'source': 'session'
        })
    
    persona = load_conversation_data(conversation_id, "persona")
    if persona:
        try:
            session_data = session.get('data', {})
            session_data['persona'] = persona
            session['data'] = session_data
        except Exception as e:
            log_error(f"Error updating session with persona: {str(e)}")
        
        return jsonify({
            'status': 'success',
            'persona': persona,
            'source': 'file'
        })
    
    return jsonify({
        'status': 'error',
        'message': 'No persona available for this conversation',
        'conversation_id': conversation_id
    })

os.makedirs("conversation_data", exist_ok=True)

def save_conversation_data(conversation_id, key, value):
    """Save a piece of conversation data to a file with improved error handling"""
    if not conversation_id:
        log_error("Cannot save data: No conversation ID provided")
        return False
        
    try:
        # Create directory for this conversation
        conversation_dir = os.path.join("conversation_data", conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)
        
        # Save data to file, handle different data types
        file_path = os.path.join(conversation_dir, f"{key}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(value, str):
                # For string values, write directly
                f.write(value)
            else:
                # For dictionaries, lists, etc. use JSON
                try:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                except:
                    # Fallback to string representation if JSON fails
                    f.write(str(value))
        
        # Log successful save
        log_debug(f"Successfully saved {key} data for conversation {conversation_id}")
        return True
        
    except Exception as e:
        error_msg = f"Error saving {key} for conversation {conversation_id}: {str(e)}"
        log_error(error_msg)
        return False

def load_conversation_data(conversation_id, key):
    """Load a piece of conversation data from a file with improved error handling"""
    if not conversation_id:
        log_error("Cannot load data: No conversation ID provided")
        return None
        
    try:
        file_path = os.path.join("conversation_data", conversation_id, f"{key}.json")
        
        if not os.path.exists(file_path):
            # Also try with .txt extension
            txt_file_path = os.path.join("conversation_data", conversation_id, f"{key}.txt")
            if os.path.exists(txt_file_path):
                file_path = txt_file_path
            else:
                return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # If empty file, return empty dict
            if not content.strip():
                return {}
                
            try:
                # Try to parse as JSON first
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return as string
                return content
                
    except Exception as e:
        error_msg = f"Error loading {key} for conversation {conversation_id}: {str(e)}"
        log_error(error_msg)
        return None
    

@app.route('/check_webhook', methods=['GET'])
def check_webhook():
    """Test endpoint to check if websockets are working"""
    try:
        conversation_id = request.args.get('conversation_id', 'test')
        message = request.args.get('message', 'Test message')
        
        # Try to emit a test event
        socketio.emit('test_event', {
            'conversation_id': conversation_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Also try to emit to different namespaces to debug
        socketio.emit('log_message', {
            'type': 'debug',
            'message': f"Test webhook: {message}"
        })
        
        return jsonify({
            'status': 'success',
            'message': 'Webhook test initiated',
            'conversation_id': conversation_id,
            'test_message': message
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error testing webhook: {str(e)}',
            'error': str(e)
        })
    
@app.route('/get_ner_results', methods=['GET'])
def get_ner_results():
    """Get the NER results for a conversation"""
    conversation_id = request.args.get('conversation_id')
    
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    # Try to get NER results from file
    ner_results = load_conversation_data(conversation_id, "ner_results")
    
    if ner_results:
        return jsonify({
            'status': 'success',
            'results': ner_results
        })
    
    # If we don't have cached results, get conversation and extract NER
    try:
        # Try to get conversation from session
        session_data = session.get('data', {})
        conversation = []
        
        if session_data.get('conversation_id') == conversation_id:
            conversation = session_data.get('conversation', [])
        
        # If not in session, try file
        if not conversation:
            conversation = load_conversation_data(conversation_id, "conversation")
        
        if not conversation:
            return jsonify({
                'status': 'error',
                'message': 'No conversation data found'
            })
        
        # Extract NER
        ner_results = extract_ner_from_conversation(conversation)
        
        # Save for future use
        save_conversation_data(conversation_id, "ner_results", ner_results)
        
        return jsonify({
            'status': 'success',
            'results': ner_results,
            'source': 'extracted'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting NER results: {str(e)}'
        })

# Fix 9: Add a function to get search results directly via API
@app.route('/get_search_results', methods=['GET'])
def get_search_results():
    """Get the most recent search results for a conversation"""
    conversation_id = request.args.get('conversation_id')
    
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    # Try to get search results from file
    search_results = load_conversation_data(conversation_id, "search_results")
    
    if search_results:
        return jsonify({
            'status': 'success',
            'results': search_results
        })
    
    # If we don't have cached results, check the search history file
    try:
        try:
            with open('search_history.json', 'r', encoding='utf-8') as f:
                search_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            search_history = []
            
        # Find the most recent search for this conversation
        relevant_searches = [
            search for search in search_history 
            if search.get('conversation_id') == conversation_id
        ]
        
        if relevant_searches:
            # Sort by timestamp if available
            if 'timestamp' in relevant_searches[0]:
                relevant_searches.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            most_recent = relevant_searches[0]
            
            # Save for future use
            save_conversation_data(conversation_id, "search_results", most_recent)
            
            return jsonify({
                'status': 'success',
                'results': most_recent,
                'source': 'history'
            })
        
        return jsonify({
            'status': 'warning',
            'message': 'No search results found for this conversation'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting search results: {str(e)}'
        })

# Fix 10: Add feedback endpoint to help debug frontend-backend communication
@app.route('/feedback', methods=['POST'])
def send_feedback():
    """Endpoint for the frontend to send feedback or debugging information"""
    data = request.get_json()
    message = data.get('message', '')
    type = data.get('type', 'debug')
    conversation_id = data.get('conversation_id')
    
    # Log the feedback
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] FEEDBACK ({type}): {message}\n"
    
    with open("logs/feedback.txt", "a", encoding="utf-8") as f:
        f.write(log_line)
    
    # Also emit as a log message
    socketio.emit('log_message', {
        'type': type,
        'message': f"Feedback: {message}"
    })
    
    # Save to conversation data if conversation_id provided
    if conversation_id:
        try:
            feedback_data = load_conversation_data(conversation_id, "feedback") or []
            feedback_data.append({
                'timestamp': timestamp,
                'type': type,
                'message': message
            })
            save_conversation_data(conversation_id, "feedback", feedback_data)
        except Exception as e:
            log_error(f"Error saving feedback: {str(e)}")
    
    return jsonify({
        'status': 'success',
        'message': 'Feedback received'
    })

@app.route('/save_conversation_direct', methods=['POST'])
def save_conversation_direct():
    """Save conversation directly from UI data"""
    data = request.get_json()
    name = data.get('name', f"Conversation-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    conversation_id = data.get('conversation_id')
    conversation = data.get('conversation', [])
    
    if not conversation:
        return jsonify({
            'status': 'error',
            'message': 'No conversation data provided'
        })
    
    # Save conversation to file
    os.makedirs("saved_conversations", exist_ok=True)
    conv_filename = f"saved_conversations/{name}_ui_saved.json"
    
    try:
        with open(conv_filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        # Also try to update session with this conversation
        try:
            session_data = session.get('data', {})
            session_data['conversation'] = conversation
            session['data'] = session_data
        except Exception as e:
            log_error(f"Error updating session with direct saved conversation: {str(e)}")
        
        return jsonify({
            'status': 'success',
            'message': f'Conversation saved directly to {conv_filename}',
            'filename': conv_filename
        })
    except Exception as e:
        error_msg = f"Error saving conversation directly: {str(e)}"
        log_error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg
        })
    
# Add this to enable session restoration
@app.before_request
def restore_session_from_params():
    """Attempt to restore session from URL parameters"""
    if request.endpoint != 'static':
        try:
            conversation_id = request.args.get('conversation_id')
            if conversation_id and 'data' not in session:
                # If we have a conversation ID but no session, try to restore it
                conversation_dir = os.path.join("conversation_data", conversation_id)
                if os.path.exists(conversation_dir):
                    # Load data from files
                    conversation = load_conversation_data(conversation_id, "conversation")
                    persona = load_conversation_data(conversation_id, "persona")
                    requirements = load_conversation_data(conversation_id, "requirements")
                    
                    if conversation or persona or requirements:
                        # Create session data
                        session_data = {
                            'conversation_id': conversation_id,
                            'conversation': conversation or [],
                            'persona': persona,
                            'requirements': requirements,
                            'previous_data': {
                                'preferences': {},
                                'num_matches': 100  # Default
                            },
                            'saved_conversations': []
                        }
                        
                        # Update session
                        session['data'] = session_data
                        log_debug(f"Restored session from URL parameter for conversation_id: {conversation_id}")
        except Exception as e:
            log_error(f"Error in session restoration from URL params: {str(e)}")


@celery.task(bind=True)
def get_openai_completion_task(self, prompt, model="o3-mini", conversation_id=None):
    """Celery task for OpenAI completion with file-based storage"""
    result = get_openai_completion(prompt, model)
    
    if conversation_id:
        socketio.emit('model_response', {
            'model': f'openai_{model}',
            'conversation_id': conversation_id,
            'response': result
        })
        
        if 'persona.txt' in prompt and result:
            try:
                session_data = session.get('data', {})
                session_data['persona'] = result
                session_data['conversation_id'] = conversation_id
                session['data'] = session_data
            except Exception as e:
                log_error(f"Error storing persona in session: {str(e)}")
            
            save_conversation_data(conversation_id, "persona", result)
            log_debug(f"Stored persona in file for conversation {conversation_id}")
        
        if 'requirement.txt' in prompt and result:
            try:
                session_data = session.get('data', {})
                session_data['requirements'] = result
                session['data'] = session_data
            except Exception as e:
                log_error(f"Error storing requirements in session: {str(e)}")
            
            save_conversation_data(conversation_id, "requirements", result)
            log_debug(f"Stored requirements in file for conversation {conversation_id}")
        
        log_debug(f"Emitted model_response event for {model} with conversation_id {conversation_id}")
    
    return result

@app.route('/store_result_directly', methods=['POST'])
def store_result_directly():
    """Store a result directly for a conversation"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    key = data.get('key')
    value = data.get('value')
    
    if not all([conversation_id, key, value]):
        return jsonify({
            'status': 'error',
            'message': 'Missing required parameters'
        })
    
    try:
        session_data = session.get('data', {})
        session_data[key] = value
        session['data'] = session_data
        
        success = save_conversation_data(conversation_id, key, value)
        
        return jsonify({
            'status': 'success',
            'message': f'Stored {key} for conversation {conversation_id}',
            'file_storage_success': success
        })
    except Exception as e:
        log_error(f"Error storing {key} for conversation {conversation_id}: {str(e)}")
        
        success = save_conversation_data(conversation_id, key, value)
        
        if success:
            return jsonify({
                'status': 'partial_success',
                'message': f'Stored {key} in file only for conversation {conversation_id}',
                'error': str(e)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to store {key} for conversation {conversation_id}',
                'error': str(e)
            })
        
@app.route('/get_requirements', methods=['GET'])
def get_requirements():
    """Get the stored requirements for a conversation"""
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    session_data = session.get('data', {})
    if session_data.get('conversation_id') == conversation_id and session_data.get('requirements'):
        return jsonify({
            'status': 'success',
            'requirements': session_data['requirements'],
            'source': 'session'
        })
    
    requirements = load_conversation_data(conversation_id, "requirements")
    if requirements:
        return jsonify({
            'status': 'success',
            'requirements': requirements,
            'source': 'file'
        })
    
    return jsonify({
        'status': 'error',
        'message': 'No requirements available for this conversation'
    })

@app.route('/get_session_data', methods=['GET'])
def get_session_data():
    """Debug endpoint to get current session data"""
    conversation_id = request.args.get('conversation_id')
    
    session_data = session.get('data', {})
    
    if conversation_id and session_data.get('conversation_id') != conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'Conversation ID mismatch',
            'requested_id': conversation_id,
            'session_id': session_data.get('conversation_id')
        })
    
    sanitized_data = {
        'conversation_id': session_data.get('conversation_id'),
        'has_persona': session_data.get('persona') is not None,
        'has_requirements': session_data.get('requirements') is not None,
        'conversation_length': len(session_data.get('conversation', [])),
        'random_number': session_data.get('random_number')
    }
    
    return jsonify({
        'status': 'success',
        'session_data': sanitized_data
    })

@app.route('/get_last_message', methods=['GET'])
def get_last_message():
    """Get the last message from a conversation"""
    conversation_id = request.args.get('conversation_id')
    message_type = request.args.get('type', 'user')
    
    session_data = session.get('data', {})
    
    if session_data.get('conversation_id') == conversation_id and session_data.get('conversation'):
        conversation = session_data.get('conversation', [])
        
        for msg in reversed(conversation):
            if msg.get('role') == message_type:
                return jsonify({
                    'status': 'success',
                    'message': msg.get('content', '')
                })
        
        return jsonify({
            'status': 'error',
            'message': f'No {message_type} message found in conversation'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No conversation found with this ID'
        })
    
@app.route('/debug', methods=['GET'])
def debug():
    """Debug route to check if server is running"""
    return jsonify({
        'status': 'success',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })
@app.route('/get_conversation', methods=['POST'])
def get_conversation():
    """Get the full conversation data"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    # Try to get from session first
    conversation = None
    try:
        session_data = session.get('data', {})
        if session_data.get('conversation_id') == conversation_id:
            conversation = session_data.get('conversation', [])
    except Exception as e:
        log_error(f"Error getting conversation from session: {str(e)}")
    
    # If not in session, try file storage
    if not conversation:
        try:
            conversation = load_conversation_data(conversation_id, "conversation")
        except Exception as e:
            log_error(f"Error loading conversation from file: {str(e)}")
    
    if conversation:
        # Create simplified conversation for UI
        simplified_conversation = []
        for msg in conversation:
            if isinstance(msg, dict) and 'role' in msg:
                simplified_conversation.append({
                    'role': msg['role'],
                    'content': msg.get('content', '')
                })
        
        return jsonify({
            'status': 'success',
            'conversation': simplified_conversation
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Conversation not found'
        })
def get_combined_critique(original_prompt, conversation_history, search_record, assistant_response):
    """
    Get critiques from Together API.
    Returns a dictionary with the critique.
    """
    together_critique = get_critic_evaluation_together(
        original_prompt,
        conversation_history,
        search_record,
        assistant_response
    )
    
    combined_critique = {
        "together": together_critique,
    }
    
    log_debug(f"Combined critique: {json.dumps(combined_critique, ensure_ascii=False)}")
    
    return combined_critique

def process_search_results(search_record):
    """
    Process search results and determine if they should be shown to the actor.
    Returns a tuple of (show_results_to_actor, search_text)
    """
    if not search_record:
        return False, ""
        
    try:
        search_response = search_record.get("results", "")
        if not search_response:
            log_debug("Empty search results found")
            return False, ""
            
        num_matches = None
        patterns = [
            r'"Number of matches":\s*(\d+)',
            r'Number of matches:\s*(\d+)',
            r'Found (\d+) matches',
            r'(\d+) results found',
            r'(\d+) hotels match'
        ]
        
        for pattern in patterns:
            matches_match = re.search(pattern, search_response, re.IGNORECASE)
            if matches_match:
                try:
                    num_matches = int(matches_match.group(1))
                    log_debug(f"Found {num_matches} matches in search results using pattern: {pattern}")
                    break
                except ValueError:
                    continue
        
        if num_matches is None:
            no_matches_patterns = [
                r'no matches',
                r'no results',
                r'0 matches',
                r'0 results'
            ]
            
            for pattern in no_matches_patterns:
                if re.search(pattern, search_response, re.IGNORECASE):
                    log_debug("Search explicitly mentions no matches")
                    num_matches = 0
                    break
        
        if num_matches is None:
            hotel_name_count = len(re.findall(r'Hotel name:', search_response, re.IGNORECASE))
            if hotel_name_count > 0:
                num_matches = hotel_name_count
                log_debug(f"Fallback: Estimated {num_matches} matches by counting 'Hotel name:' occurrences")
            else:
                line_count = len([line for line in search_response.split('\n') if line.strip()])
                num_matches = line_count
                log_debug(f"Last resort: Setting matches to line count: {num_matches}")
        
        if num_matches is None:
            log_debug("Could not determine number of matches, defaulting to 100")
            num_matches = 100
        
        search_record["num_matches"] = num_matches
        
        log_debug(f"Search result display decision: num_matches={num_matches}, threshold=50, comparison result: {num_matches < 50}")
        
        if num_matches < 50:
            log_debug(f"SHOWING search results to actor ({num_matches} matches < 50)")
            return True, search_response
        else:
            log_debug(f"NOT showing search results to actor ({num_matches} matches >= 50)")
            return False, ""
            
    except Exception as e:
        error_msg = f"Error processing search results: {str(e)}"
        log_error(error_msg)
        return False, ""
    
@app.route('/restore_session', methods=['POST'])
def restore_session():
    """Restore session data from file storage"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({
            'status': 'error',
            'message': 'No conversation ID provided'
        })
    
    try:
        # Load data from files
        conversation = load_conversation_data(conversation_id, "conversation")
        persona = load_conversation_data(conversation_id, "persona")
        requirements = load_conversation_data(conversation_id, "requirements")
        ner_results = load_conversation_data(conversation_id, "ner_results")
        saved_conversations = load_conversation_data(conversation_id, "saved_conversations")
        
        # Create session data
        session_data = {
            'conversation_id': conversation_id,
            'conversation': conversation or [],
            'persona': persona,
            'requirements': requirements,
            'previous_data': {
                'preferences': ner_results or {},
                'num_matches': 100  # Default
            },
            'saved_conversations': saved_conversations or []
        }
        
        # Update session
        session['data'] = session_data
        
        return jsonify({
            'status': 'success',
            'message': 'Session restored from file storage'
        })
    except Exception as e:
        error_msg = f"Error restoring session: {str(e)}"
        log_error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg
        })

# Add this to enable session restoration
@app.before_request
def restore_session_from_params():
    """Attempt to restore session from URL parameters"""
    if request.endpoint != 'static':
        try:
            conversation_id = request.args.get('conversation_id')
            if conversation_id and 'data' not in session:
                # If we have a conversation ID but no session, try to restore it
                conversation_dir = os.path.join("conversation_data", conversation_id)
                if os.path.exists(conversation_dir):
                    # Load data from files
                    conversation = load_conversation_data(conversation_id, "conversation")
                    persona = load_conversation_data(conversation_id, "persona")
                    requirements = load_conversation_data(conversation_id, "requirements")
                    
                    if conversation or persona or requirements:
                        # Create session data
                        session_data = {
                            'conversation_id': conversation_id,
                            'conversation': conversation or [],
                            'persona': persona,
                            'requirements': requirements,
                            'previous_data': {
                                'preferences': {},
                                'num_matches': 100  # Default
                            },
                            'saved_conversations': []
                        }
                        
                        # Update session
                        session['data'] = session_data
                        log_debug(f"Restored session from URL parameter for conversation_id: {conversation_id}")
        except Exception as e:
            log_error(f"Error in session restoration from URL params: {str(e)}")

            
if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)



