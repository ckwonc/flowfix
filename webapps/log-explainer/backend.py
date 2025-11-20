from dataiku.customwebapp import *

# Access the parameters that end-users filled in using webapp config
# For example, for a parameter called "input_dataset"
# input_dataset = get_webapp_config()["input_dataset"]

"""
AI Log Assistant - Backend with LogAI Integration
Dataiku Webapp Backend for Log Analysis

This backend provides endpoints to:
1. Fetch recent Dataiku jobs
2. Get full logs for a specific job
3. Analyze logs with AI using LogAI preprocessing (Phase 3)
"""

import dataiku
from flask import request, jsonify
import json
import datetime
import logging
import pandas as pd
import re

# LogAI imports
from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dataiku client
client = dataiku.api_client()
project = client.get_default_project()


# ============================================================================
# LogAI Setup and Functions
# ============================================================================

def create_logai_preprocessor():
    """
    Create LogAI preprocessor configured for Dataiku logs
    Replaces common patterns with tokens for better parsing
    """
    config = PreprocessorConfig(
        custom_replace_list=[
            # Dataiku-specific patterns
            [r"blk_[-\d]+", "<BLOCK_ID>"],                    # Block IDs
            [r"\d+\.\d+\.\d+\.\d+", "<IP>"],                 # IP addresses  
            [r"(/[\w\-./]+)+", "<FILE_PATH>"],               # File paths
            [r"\d{4}-\d{2}-\d{2}", "<DATE>"],                # Dates (YYYY-MM-DD)
            [r"\d{2}:\d{2}:\d{2}[.,]\d+", "<TIMESTAMP>"],    # Timestamps with ms
            [r"\d{2}:\d{2}:\d{2}", "<TIME>"],                # Time (HH:MM:SS)
            [r"line \d+", "line <LINE_NUM>"],                # Line numbers in errors
            [r"'[a-zA-Z_]\w*'", "'<VAR_NAME>'"],            # Variable names in quotes
            [r'"[a-zA-Z_]\w*"', '"<VAR_NAME>"'],            # Variable names in double quotes
            [r"0x[0-9a-fA-F]+", "<HEX>"],                    # Hex values
            [r"\b\d{4,}\b", "<LONG_NUM>"],                   # Long numbers (4+ digits)
            [r"\b\d+\b", "<NUM>"],                           # Other numbers
        ]
    )
    return Preprocessor(config)


def create_logai_parser():
    """
    Create LogAI parser using Drain algorithm
    Extracts log templates from unstructured text
    """
    parsing_params = DrainParams(
        sim_th=0.5,      # Similarity threshold (0.4-0.6 recommended)
        depth=5          # Parse tree depth (4-6 recommended)
    )
    
    config = LogParserConfig(
        parsing_algorithm="drain",
        parsing_algo_params=parsing_params
    )
    
    return LogParser(config)


def process_log_with_logai(log_content):
    """
    Process Dataiku log using LogAI to extract structured information
    
    Args:
        log_content: Raw log text from Dataiku job
        
    Returns:
        dict with structured log information:
        {
            'error_template': str,       # Extracted error pattern template
            'error_line': str,           # Original error line from log
            'error_line_number': int,    # Line number where error occurred
            'all_error_lines': list,     # All error-related lines
            'log_patterns': list,        # Top unique log patterns found
            'cleaned_log': str,          # Preprocessed log
            'summary': dict              # Summary statistics
        }
    """
    try:
        logger.info("Processing log with LogAI...")
        
        # Split log into lines
        log_lines = log_content.split('\n')
        if not log_lines:
            logger.warning("Empty log content")
            return _empty_logai_result()
        
        # Step 1: Preprocess logs
        logger.info(f"Preprocessing {len(log_lines)} log lines...")
        preprocessor = create_logai_preprocessor()
        cleaned_lines, custom_patterns = preprocessor.clean_log(pd.Series(log_lines))
        
        # Step 2: Parse logs to extract templates using Drain
        logger.info("Parsing logs with Drain algorithm to extract templates...")
        parser = create_logai_parser()
        parsed_result = parser.parse(cleaned_lines)
        
        # Step 3: Extract error information
        error_info = _extract_error_info(
            log_lines, 
            cleaned_lines, 
            parsed_result
        )
        
        logger.info(f"LogAI processing complete. Found {error_info['summary']['error_line_count']} error lines")
        
        return error_info
        
    except Exception as e:
        logger.error(f"Error in LogAI processing: {str(e)}", exc_info=True)
        # Return empty result on failure so analysis can continue
        return _empty_logai_result()


def _extract_error_info(original_lines, cleaned_lines, parsed_result):
    """
    Extract structured error information from parsed logs
    
    Intelligently identifies actual recipe errors (Python or SQL) while
    filtering out Java infrastructure noise. Focuses on user-relevant errors.
    """
    
    # Step 1: Find Python recipe error section
    python_error_section = _extract_python_error_section(original_lines)
    
    # Step 2: Find SQL error section
    sql_error_section = _extract_sql_error_section(original_lines)
    
    # Step 3: Determine which error is primary
    if python_error_section:
        primary_error = python_error_section
        error_type = 'Python Recipe Error'
    elif sql_error_section:
        primary_error = sql_error_section
        error_type = 'SQL Recipe Error'
    else:
        # Fallback to generic error detection
        primary_error = _extract_generic_error(original_lines, cleaned_lines, parsed_result)
        error_type = 'Generic Error'
    
    # Get unique log patterns
    unique_patterns = parsed_result['parsed_logline'].unique().tolist()
    
    # Create summary statistics
    summary = {
        'total_lines': len(original_lines),
        'unique_patterns': len(unique_patterns),
        'error_line_count': len(primary_error.get('all_lines', [])),
        'has_errors': primary_error is not None,
        'error_type': error_type
    }
    
    # Build structured result
    return {
        'error_template': primary_error.get('template'),
        'error_line': primary_error.get('message'),
        'error_line_number': primary_error.get('line_number'),
        'all_error_lines': primary_error.get('all_lines', []),
        'log_patterns': unique_patterns[:10],
        'cleaned_log': '\n'.join(cleaned_lines.tolist()),
        'summary': summary,
        'error_context': primary_error.get('context', '')
    }


def _extract_python_error_section(lines):
    """
    Extract Python recipe error from Dataiku logs
    
    Looks for the pattern:
    *************** Recipe code failed **************
    Begin Python stack
    <actual error>
    End Python stack
    """
    error_info = None
    in_python_error = False
    python_stack_lines = []
    
    for idx, line in enumerate(lines):
        # Start of Python error section
        if 'Recipe code failed' in line or 'Begin Python stack' in line:
            in_python_error = True
            continue
        
        # End of Python error section
        if 'End Python stack' in line:
            if python_stack_lines:
                # Parse the stack to extract error details
                error_info = _parse_python_stack(python_stack_lines, idx)
            break
        
        # Collect lines in the Python error section
        if in_python_error:
            python_stack_lines.append({
                'line_number': idx + 1,
                'text': line.strip()
            })
    
    return error_info


def _parse_python_stack(stack_lines, section_end_line):
    """
    Parse Python stack trace to extract meaningful error information
    
    Example:
    Traceback (most recent call last):
      File "/opt/dataiku/python/dataiku/container/exec_py_recipe.py", line 15
        exec(fd.read())
      File "<string>", line 20, in <module>
    NameError: name 'test_df' is not defined
    """
    error_type = None
    error_message = None
    error_line_number = None
    traceback_lines = []
    
    for item in stack_lines:
        line = item['text']
        
        # Extract error type and message (usually last line)
        # Format: ErrorType: error message
        if ':' in line and not line.strip().startswith('File'):
            # Check if this looks like an error (starts with capital letter, has colon)
            parts = line.split(':', 1)
            if parts[0].strip() and parts[0].strip()[0].isupper():
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else ''
        
        # Extract line number from traceback
        # Format: File "<string>", line 20, in <module>
        if 'File "<string>"' in line and 'line' in line:
            match = re.search(r'line (\d+)', line)
            if match:
                error_line_number = int(match.group(1))
        
        traceback_lines.append(line)
    
    # Build error template
    template = f"{error_type}: {error_message}" if error_type else error_message
    
    return {
        'template': template,
        'message': error_message,
        'error_type': error_type,
        'line_number': error_line_number,
        'all_lines': stack_lines,
        'context': '\n'.join(traceback_lines)
    }


def _extract_sql_error_section(lines):
    """
    Extract SQL recipe error from Dataiku logs
    
    Looks for SQL errors like:
    - SQL execution error
    - Syntax error at or near
    - Column does not exist
    - PSQLException
    """
    sql_error_keywords = [
        'sql execution error',
        'syntax error',
        'psqlexception',
        'sqlexception',
        'column does not exist',
        'table does not exist',
        'relation does not exist',
        'invalid sql',
        'sql error'
    ]
    
    error_lines = []
    error_message = None
    error_context = []
    
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if this line contains a SQL error
        if any(keyword in line_lower for keyword in sql_error_keywords):
            error_lines.append({
                'line_number': idx + 1,
                'text': line.strip()
            })
            
            # Try to extract the actual error message
            if not error_message:
                # SQL errors often have format: "ERROR: <message>"
                if 'error:' in line_lower:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        error_message = parts[1].strip()
                else:
                    error_message = line.strip()
            
            # Collect context (a few lines around the error)
            start = max(0, idx - 2)
            end = min(len(lines), idx + 3)
            error_context = lines[start:end]
    
    if not error_lines:
        return None
    
    # Build error template (clean up common SQL error patterns)
    template = error_message if error_message else "SQL Error"
    
    # Try to extract more specific info
    error_type = "SQL Error"
    if 'syntax error' in template.lower():
        error_type = "SQL Syntax Error"
    elif 'does not exist' in template.lower():
        error_type = "SQL Object Not Found"
    
    return {
        'template': f"{error_type}: {template}",
        'message': error_message,
        'error_type': error_type,
        'line_number': error_lines[0]['line_number'] if error_lines else None,
        'all_lines': error_lines,
        'context': '\n'.join(error_context)
    }


def _extract_generic_error(original_lines, cleaned_lines, parsed_result):
    """
    Fallback: Generic error extraction for other error types
    """
    error_keywords = [
        'error', 'exception', 'failed', 'failure', 'traceback'
    ]
    
    # Avoid infrastructure noise
    ignore_keywords = [
        'com.dataiku',  # Java package names
        'at com.',
        'at java.',
        'kubectl',       # Kubernetes noise
        'container'
    ]
    
    error_lines = []
    
    # Scan for error lines, but filter out infrastructure noise
    for idx, (original, cleaned, template) in enumerate(
        zip(original_lines, cleaned_lines, parsed_result['parsed_logline'])
    ):
        line_lower = original.lower()
        
        # Check if this is an error line
        has_error = any(keyword in line_lower for keyword in error_keywords)
        is_infrastructure = any(keyword in original for keyword in ignore_keywords)
        
        if has_error and not is_infrastructure:
            error_lines.append({
                'line_number': idx + 1,
                'original': original,
                'cleaned': cleaned,
                'template': template
            })
    
    if not error_lines:
        return {
            'template': None,
            'message': None,
            'line_number': None,
            'all_lines': [],
            'context': ''
        }
    
    # Use the last meaningful error
    primary_error = error_lines[-1]
    
    return {
        'template': primary_error['template'],
        'message': primary_error['original'],
        'error_type': 'Generic',
        'line_number': primary_error['line_number'],
        'all_lines': error_lines,
        'context': primary_error['original']
    }


def _empty_logai_result():
    """Return empty LogAI result structure when processing fails"""
    return {
        'error_template': None,
        'error_line': None,
        'error_line_number': None,
        'all_error_lines': [],
        'log_patterns': [],
        'cleaned_log': '',
        'summary': {
            'total_lines': 0,
            'unique_patterns': 0,
            'error_line_count': 0,
            'has_errors': False,
            'error_type': 'Unknown'
        },
        'error_context': ''
    }


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/api/jobs')
def get_jobs():
    """
    Fetch recent Dataiku jobs with filtering
    
    Query params:
        - filter: 'all', 'failed', or 'success' (default: 'failed')
        - limit: max number of jobs to return (default: 20)
    
    Returns:
        JSON with list of jobs
    """
    try:
        filter_status = request.args.get('filter', 'failed')
        limit = int(request.args.get('limit', 20))
        
        logger.info(f"Fetching jobs with filter={filter_status}, limit={limit}")
        
        # Get all jobs from project
        all_jobs = project.list_jobs()
        
        # Filter jobs based on status
        filtered_jobs = []
        for job in all_jobs:
            job_state = job.get('state', '')
            
            # Apply filter
            if filter_status == 'all':
                include = True
            elif filter_status == 'failed':
                include = (job_state == 'FAILED')
            elif filter_status == 'success':
                include = (job_state == 'DONE')
            else:
                include = True
            
            if include:
                # Extract relevant job information
                job_def = job.get('def', {})
                job_id = job_def.get('id', '')
                job_name = job_def.get('name', 'Unknown Job')
                timestamp_ms = job_def.get('initiationTimestamp', 0)
                
                # Convert timestamp to readable format
                if timestamp_ms:
                    timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
                else:
                    timestamp_str = 'Unknown'
                
                # Determine status
                if job_state == 'FAILED':
                    status = 'failed'
                elif job_state == 'DONE':
                    status = 'success'
                else:
                    status = job_state.lower()
                
                # Get basic error info (we'll get full log later)
                error_message = None
                if status == 'failed':
                    error_message = f"{job_state} - See logs for details"
                
                filtered_jobs.append({
                    'id': job_id,
                    'name': job_name,
                    'status': status,
                    'timestamp': timestamp_str,
                    'error': error_message,
                    'fullError': None  # Will be populated when user clicks Analyze
                })
        
        # Sort by timestamp (most recent first) and limit
        filtered_jobs.sort(key=lambda x: x['timestamp'], reverse=True)
        filtered_jobs = filtered_jobs[:limit]
        
        logger.info(f"Returning {len(filtered_jobs)} jobs")
        
        return jsonify({
            'success': True,
            'jobs': filtered_jobs,
            'count': len(filtered_jobs)
        })
        
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/jobs/<job_id>/logs')
def get_job_logs(job_id):
    """
    Fetch full log content for a specific job
    
    Args:
        job_id: The Dataiku job ID (from URL path)
    
    Returns:
        JSON with job details and full log content
    """
    try:
        logger.info(f"Fetching logs for job: {job_id}")
        
        # Get job handle
        job = project.get_job(job_id)
        
        # Get full log content
        log_content = job.get_log()
        
        # Get job details
        all_jobs = project.list_jobs()
        job_info = next((j for j in all_jobs if j['def']['id'] == job_id), None)
        
        if job_info:
            job_def = job_info.get('def', {})
            metadata = {
                'name': job_def.get('name', 'Unknown'),
                'state': job_info.get('state', 'UNKNOWN'),
                'initiationTimestamp': job_def.get('initiationTimestamp', 0)
            }
        else:
            metadata = {}
        
        logger.info(f"Retrieved {len(log_content)} characters of log data")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'logs': log_content,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Error fetching logs for job {job_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_logs():
    """
    Analyze logs with AI using LLM Mesh + LogAI preprocessing
    
    Request body:
        {
            "job_id": "string",
            "llm_id": "string"  # Full LLM ID like "openai:connection:model"
        }
    
    Returns:
        JSON with structured analysis results including LogAI insights
    """
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        llm_id = data.get('llm_id')
        
        if not job_id or not llm_id:
            return jsonify({
                'success': False,
                'error': 'Missing job_id or llm_id'
            }), 400
        
        logger.info(f"Analyzing job {job_id} with LLM {llm_id}")
        
        # Step 1: Get the full log content
        job = project.get_job(job_id)
        log_content = job.get_log()
        
        # Get job metadata for context
        all_jobs = project.list_jobs()
        job_info = next((j for j in all_jobs if j['def']['id'] == job_id), None)
        
        job_name = "Unknown Job"
        job_type = "Unknown"
        recipe_name = None
        input_datasets = []
        output_datasets = []
        
        if job_info:
            job_def = job_info.get('def', {})
            job_name = job_def.get('name', 'Unknown Job')
            job_type = job_def.get('type', 'Unknown')
            recipe_name = job_def.get('recipe', None)
            
            # Extract input/output dataset information
            outputs = job_def.get('outputs', [])
            for output in outputs:
                if output.get('type') == 'DATASET':
                    dataset_name = output.get('targetDataset', 'Unknown')
                    output_datasets.append(dataset_name)
            
            # Get inputs from recipe
            if recipe_name:
                try:
                    recipe = project.get_recipe(recipe_name)
                    recipe_def = recipe.get_definition()
                    inputs = recipe_def.get('inputs', {})
                    
                    # Iterate through all input roles
                    for role_name, role_data in inputs.items():
                        items = role_data.get('items', [])
                        for item in items:
                            ref = item.get('ref', '')
                            # Remove project key prefix if present
                            if '.' in ref:
                                ref = ref.split('.', 1)[1]
                            input_datasets.append(ref)
                except Exception as e:
                    logger.warning(f"Could not fetch recipe inputs: {str(e)}")
        
        logger.info(f"Retrieved {len(log_content)} characters of log data")
        logger.info(f"Job context - Type: {job_type}, Recipe: {recipe_name}")
        
        # Step 2: Process log with LogAI
        logger.info("=" * 60)
        logger.info("PHASE 3: Processing with LogAI")
        logger.info("=" * 60)
        
        logai_analysis = process_log_with_logai(log_content)
        
        logger.info(f"LogAI extracted error type: {logai_analysis['summary'].get('error_type')}")
        logger.info(f"Error template: {logai_analysis.get('error_template')}")
        logger.info(f"Error at line: {logai_analysis.get('error_line_number')}")
        
        # Use error context instead of full log for more focused LLM analysis
        # This gives the LLM just the relevant error section
        error_context = logai_analysis.get('error_context', '')
        
        # If we have error context, use it; otherwise fall back to cleaned log
        if error_context and len(error_context) > 50:
            processed_log = error_context
            logger.info("Using extracted error context for LLM analysis")
        else:
            processed_log = logai_analysis['cleaned_log']
            logger.info("Using full cleaned log for LLM analysis")
            
            # Truncate if too long
            max_log_length = 8000
            if len(processed_log) > max_log_length:
                # Get the last portion (errors usually at the end)
                processed_log = "...[log truncated]...\n\n" + processed_log[-max_log_length:]
        
        # Step 3: Build enhanced context with LogAI insights
        context_info = f"""
Job Name: {job_name}
Job Type: {job_type}
Recipe: {recipe_name if recipe_name else 'N/A'}
Input Datasets: {', '.join(input_datasets) if input_datasets else 'N/A'}
Output Datasets: {', '.join(output_datasets) if output_datasets else 'N/A'}
"""
        
        # LogAI insights for context
        logai_context = f"""
LogAI Analysis Summary:
- Error Type: {logai_analysis['summary']['error_type']}
- Total Log Lines: {logai_analysis['summary']['total_lines']}
- Unique Log Patterns: {logai_analysis['summary']['unique_patterns']}
- Error Template Extracted: {logai_analysis['error_template'] or 'Not extracted'}
- Error at Line: {logai_analysis['error_line_number'] or 'Unknown'}
"""
        
        # Step 4: Prepare enhanced prompt for LLM
        system_prompt = """You are an expert analyzing Dataiku job failures. 

Provide analysis in exactly this format:

**What Happened**
State the specific error and where it occurred. Example: "NameError on line 20 - variable 'test_df' is not defined."

**Why It Happened**
Explain what caused it. Example: 'The variable test_df was referenced before being created. This means the dataset wasn't loaded or the variable name is wrong.'

**Technical Details**
Provide useful context about the job (use the context provided):
- Affected Component: What recipe, dataset, or code file was involved
- Input Data: What datasets or sources were being read
- Output Target: What was being written to
- Job Type: Type of operation (e.g., Python recipe, SQL query, visual recipe)

Format this section clearly with each item on its own line.

**How To Fix It**
List specific fixes as bullets:
• Add this line before line 20: test_df = dataiku.Dataset('dataset_name').get_dataframe()
• Check the recipe inputs to verify the dataset connection
• Ensure variable names match between steps

Keep each section under 4 sentences except How To Fix It. Be specific, not generic."""
        
        user_prompt = f"""Analyze this Dataiku job failure:

Job Context:
{context_info}

LogAI Insights:
{logai_context}

Processed Log (cleaned by LogAI):
{processed_log}

Primary Error Template Identified: {logai_analysis['error_template']}
Error Line Number: {logai_analysis['error_line_number']}
"""
        
        # Step 5: Call LLM Mesh
        logger.info(f"Calling LLM Mesh with model: {llm_id}")
        
        llm = project.get_llm(llm_id)
        completion = llm.new_completion()
        
        # Add messages to the completion
        completion.with_message(system_prompt, role="system")
        completion.with_message(user_prompt, role="user")
        
        # Execute the completion
        response = completion.execute()
        
        if not response.success:
            logger.error(f"LLM completion failed: {response}")
            return jsonify({
                'success': False,
                'error': 'LLM analysis failed'
            }), 500
        
        # Get the response text
        analysis_text = response.text
        
        logger.info("LLM analysis completed successfully")
        
        # Step 6: Parse the response into structured format
        analysis_result = parse_llm_response(analysis_text)
        
        # Step 7: Return enhanced response with LogAI insights
        return jsonify({
            'success': True,
            'job_id': job_id,
            'llm_used': llm_id,
            'analysis': analysis_result,
            'logai_insights': {
                'error_template': logai_analysis['error_template'],
                'error_line_number': logai_analysis['error_line_number'],
                'unique_patterns': logai_analysis['summary']['unique_patterns'],
                'total_lines': logai_analysis['summary']['total_lines'],
                'error_count': logai_analysis['summary']['error_line_count']
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def parse_llm_response(text):
    """
    Parse LLM response into structured format
    
    Extracts the four main sections from LLM response
    """
    sections = {
        'errorSummary': '',
        'rootCause': '',
        'stackTrace': '',
        'suggestedActions': []
    }
    
    # Simple parsing by looking for section headers
    lines = text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers
        if 'what happened' in line_lower or 'error summary' in line_lower or line_lower.startswith('1.'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'errorSummary'
            current_content = []
        elif 'why it happened' in line_lower or 'root cause' in line_lower or line_lower.startswith('2.'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'rootCause'
            current_content = []
        elif 'technical details' in line_lower or 'stack trace' in line_lower or line_lower.startswith('3.'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'stackTrace'
            current_content = []
        elif 'how to fix' in line_lower or 'suggested action' in line_lower or 'fix it' in line_lower or line_lower.startswith('4.'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'suggestedActions'
            current_content = []
        else:
            # Skip the header line itself (with ** formatting)
            if line.strip().startswith('**') and line.strip().endswith('**'):
                continue
                
            # Add content to current section
            if current_section and line.strip():
                # For suggested actions, detect bullet points or numbered items
                if current_section == 'suggestedActions':
                    cleaned_line = line.strip()
                    # Remove common bullet/number prefixes
                    for prefix in ['- ', '• ', '* ', '+ ', '→ ']:
                        if cleaned_line.startswith(prefix):
                            cleaned_line = cleaned_line[len(prefix):].strip()
                            break
                    # Remove numbered prefixes like "1. " or "1) "
                    cleaned_line = re.sub(r'^\d+[\.)]\s*', '', cleaned_line)
                    if cleaned_line:
                        sections['suggestedActions'].append(cleaned_line)
                else:
                    current_content.append(line)
    
    # Add last section
    if current_section and current_content:
        if current_section == 'suggestedActions':
            # Already handled above
            pass
        else:
            sections[current_section] = '\n'.join(current_content).strip()
    
    # If parsing failed, use the whole text
    if not sections['errorSummary'] and not sections['rootCause']:
        sections['errorSummary'] = text[:500]  # First 500 chars
        sections['rootCause'] = text
    
    return sections


@app.route('/api/llm/connections')
def get_llm_connections():
    """
    Get all available LLM connections from LLM Mesh
    
    Returns:
        JSON with list of unique LLM connections (grouped by connection name)
    """
    try:
        logger.info("Fetching LLM connections from LLM Mesh")
        
        # Get all LLMs available in the project
        all_llms = project.list_llms()
        
        # Extract unique connections
        # LLM ID format: llm_type:connection_name:model_name
        connections = {}
        
        for llm in all_llms:
            llm_id = llm.get('id', '')
            parts = llm_id.split(':')
            
            if len(parts) >= 2:
                llm_type = parts[0]  # e.g., 'openai', 'anthropic', 'azureopenai'
                connection_name = parts[1]  # e.g., 'SE-open-ai', 'my-anthropic'
                
                # Create a unique key for this connection
                connection_key = f"{llm_type}:{connection_name}"
                
                if connection_key not in connections:
                    connections[connection_key] = {
                        'id': connection_key,
                        'type': llm_type,
                        'name': connection_name,
                        'display_name': f"{connection_name} ({llm_type.upper()})",
                        'model_count': 0
                    }
                
                connections[connection_key]['model_count'] += 1
        
        # Convert to list
        connection_list = list(connections.values())
        
        logger.info(f"Found {len(connection_list)} unique LLM connections")
        
        return jsonify({
            'success': True,
            'connections': connection_list,
            'count': len(connection_list)
        })
        
    except Exception as e:
        logger.error(f"Error fetching LLM connections: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/llm/connections/<connection_id>/models')
def get_llm_models(connection_id):
    """
    Get all models available for a specific LLM connection
    
    Args:
        connection_id: Connection ID in format "llm_type:connection_name"
    
    Returns:
        JSON with list of models for this connection
    """
    try:
        logger.info(f"Fetching models for connection: {connection_id}")
        
        # Get all LLMs
        all_llms = project.list_llms()
        
        # Filter models for this connection
        models = []
        
        for llm in all_llms:
            llm_id = llm.get('id', '')
            description = llm.get('description', '')
            
            # Check if this LLM belongs to the requested connection
            if llm_id.startswith(connection_id + ':'):
                # Extract model name (everything after the connection_id)
                model_part = llm_id.replace(connection_id + ':', '', 1)
                
                # For complex IDs like "huggingfacelocal:myhf:model:TEXT_GENERATION:etc"
                # We want to show a clean model name
                model_name = model_part.split(':')[0] if ':' in model_part else model_part
                
                models.append({
                    'id': llm_id,  # Full LLM ID for API calls
                    'name': model_name,  # Clean model name for display
                    'description': description,
                    'display_name': description if description else model_name
                })
        
        logger.info(f"Found {len(models)} models for connection {connection_id}")
        
        return jsonify({
            'success': True,
            'connection_id': connection_id,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error fetching models for connection {connection_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health_check():
    """
    Health check endpoint to verify backend is running
    """
    return jsonify({
        'success': True,
        'message': 'Backend is healthy with LogAI integration',
        'project': project.project_key,
        'logai_enabled': True
    })


# For local testing/debugging
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Backend initialized successfully with LogAI")
    logger.info("=" * 60)
    logger.info(f"Project: {project.project_key}")
    logger.info("Available endpoints:")
    logger.info("  GET  /api/jobs - List recent jobs")
    logger.info("  GET  /api/jobs/<job_id>/logs - Get logs for specific job")
    logger.info("  GET  /api/llm/connections - List available LLM connections")
    logger.info("  GET  /api/llm/connections/<connection_id>/models - List models for connection")
    logger.info("  POST /api/analyze - Analyze logs with AI + LogAI preprocessing")
    logger.info("  GET  /api/health - Health check")
    logger.info("=" * 60)