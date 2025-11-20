// Access the parameters that end-users filled in using webapp config
// For example, for a parameter called "input_dataset"
// input_dataset = dataiku.getWebAppConfig()['input_dataset']

// ===========================
// AI Log Assistant - JavaScript
// ===========================

// Default LLM settings
const DEFAULT_LLM_CONNECTION = 'openai:SE-open-ai';
const DEFAULT_LLM_MODEL = 'gpt-5-mini';

// State management
let currentFilter = 'failed';
let selectedLogId = null;
let currentAnalysis = null;
let logs = []; // Store fetched logs

// DOM elements - will be initialized in init()
let logList;
let emptyState;
let analysisResults;
let loadingState;
let resultsContent;
let filterSelector;
let llmConnectionSelector;
let llmModelSelector;
let closeAnalysisBtn;
let expandStackTraceBtn;
let stackTraceContent;
let feedbackHelpful;
let feedbackNotHelpful;

// Initialize the app
function init() {
    console.log('Initializing AI Log Assistant...');
    
    // Initialize DOM elements after DOM is ready
    logList = document.getElementById('log-list');
    emptyState = document.getElementById('empty-state');
    analysisResults = document.getElementById('analysis-results');
    loadingState = document.getElementById('loading-state');
    resultsContent = document.getElementById('results-content');
    filterSelector = document.getElementById('filter-selector');
    llmConnectionSelector = document.getElementById('llm-connection-selector');
    llmModelSelector = document.getElementById('llm-model-selector');
    closeAnalysisBtn = document.getElementById('close-analysis');
    expandStackTraceBtn = document.getElementById('expand-stack-trace');
    stackTraceContent = document.getElementById('stack-trace');
    feedbackHelpful = document.getElementById('feedback-helpful');
    feedbackNotHelpful = document.getElementById('feedback-not-helpful');
    
    // Now that DOM elements are initialized, proceed with the rest
    fetchLLMConnections();
    fetchJobs();
    attachEventListeners();
}

// Fetch available LLM connections
async function fetchLLMConnections() {
    try {
        console.log('Fetching LLM connections...');
        
        const url = getWebAppBackendUrl('api/llm/connections');
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.connections.length > 0) {
            // Populate connection dropdown
            llmConnectionSelector.innerHTML = '<option value="">Select a connection...</option>';
            let defaultConnectionFound = false;
            
            data.connections.forEach(conn => {
                const option = document.createElement('option');
                option.value = conn.id;
                option.textContent = conn.display_name;
                
                // Check if this is the default connection
                if (conn.id === DEFAULT_LLM_CONNECTION) {
                    option.selected = true;
                    defaultConnectionFound = true;
                }
                
                llmConnectionSelector.appendChild(option);
            });
            
            console.log(`Loaded ${data.connections.length} LLM connections`);
            
            // If default connection was found, automatically load its models
            if (defaultConnectionFound) {
                console.log(`Auto-selecting default connection: ${DEFAULT_LLM_CONNECTION}`);
                await fetchLLMModels(DEFAULT_LLM_CONNECTION);
            }
        } else {
            llmConnectionSelector.innerHTML = '<option value="">No connections available</option>';
            console.warn('No LLM connections found');
        }
    } catch (error) {
        console.error('Error fetching LLM connections:', error);
        llmConnectionSelector.innerHTML = '<option value="">Error loading connections</option>';
    }
}

// Fetch models for selected connection
async function fetchLLMModels(connectionId) {
    try {
        console.log(`Fetching models for connection: ${connectionId}`);
        
        llmModelSelector.innerHTML = '<option value="">Loading models...</option>';
        llmModelSelector.disabled = false;
        
        const url = getWebAppBackendUrl(`api/llm/connections/${connectionId}/models`);
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.models.length > 0) {
            // Populate model dropdown
            llmModelSelector.innerHTML = '<option value="">Select a model...</option>';
            let defaultModelFound = true;
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;  // Full LLM ID
                option.textContent = model.display_name;
                
                // Check if this is the default model
                if (model.id === DEFAULT_LLM_MODEL || model.display_name === DEFAULT_LLM_MODEL) {
                    option.selected = true;
                    defaultModelFound = true;
                }
                
                llmModelSelector.appendChild(option);
            });
            
            llmModelSelector.disabled = false;
            console.log(`Loaded ${data.models.length} models`);
            
            if (defaultModelFound) {
                console.log(`Auto-selected default model: ${DEFAULT_LLM_MODEL}`);
            }
        } else {
            llmModelSelector.innerHTML = '<option value="">No models available</option>';
            console.warn('No models found for this connection');
        }
    } catch (error) {
        console.error('Error fetching models:', error);
        llmModelSelector.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Fetch jobs from backend
async function fetchJobs() {
    try {
        console.log(`Fetching jobs with filter: ${currentFilter}`);
        
        const url = getWebAppBackendUrl('api/jobs') + `?filter=${currentFilter}&limit=20`;
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            logs = data.jobs;
            console.log(`Fetched ${logs.length} jobs`);
            renderLogList();
        } else {
            console.error('Error fetching jobs:', data.error);
            showError('Failed to fetch jobs: ' + data.error);
        }
    } catch (error) {
        console.error('Error fetching jobs:', error);
        showError('Failed to connect to backend: ' + error.message);
    }
}

// Show error message
function showError(message) {
    logList.innerHTML = `
        <div style="padding: 20px; text-align: center; color: var(--text-secondary);">
            <p>${message}</p>
            <button onclick="fetchJobs()" style="margin-top: 12px; padding: 8px 16px; background: var(--accent-blue); color: white; border: none; border-radius: 6px; cursor: pointer;">
                Retry
            </button>
        </div>
    `;
}

// Render the log list based on current filter
function renderLogList() {
    logList.innerHTML = '';

    if (logs.length === 0) {
        logList.innerHTML = `
            <div style="padding: 40px 20px; text-align: center; color: var(--text-secondary);">
                <p>No ${currentFilter} jobs found</p>
            </div>
        `;
        return;
    }

    logs.forEach(log => {
        const logItem = createLogItem(log);
        logList.appendChild(logItem);
    });
}

// Create a log item element
function createLogItem(log) {
    const item = document.createElement('div');
    item.className = 'log-item';
    if (selectedLogId === log.id) {
        item.classList.add('active');
    }

    const statusClass = log.status === 'failed' ? 'failed' : 'success';
    
    item.innerHTML = `
        <div class="log-item-header">
            <div class="log-item-name">${log.name}</div>
        </div>
        <div class="log-item-meta">
            <span class="log-status ${statusClass}">${log.status}</span>
            <span class="log-timestamp">${log.timestamp}</span>
        </div>
        ${log.error ? `<div class="log-error">${log.error}</div>` : ''}
        <div class="log-actions">
            <button class="analyze-button" data-log-id="${log.id}" ${log.status === 'success' ? 'disabled' : ''}>
                Analyze
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
            </button>
        </div>
    `;

    // Add click handler for analyze button
    const analyzeBtn = item.querySelector('.analyze-button');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            handleAnalyze(log.id);
        });
    }

    return item;
}

// Handle analyze button click
async function handleAnalyze(logId) {
    const log = logs.find(l => l.id === logId);
    if (!log) return;

    // Check if LLM model is selected
    const selectedLLMId = llmModelSelector.value;
    if (!selectedLLMId) {
        alert('Please select an LLM connection and model first');
        return;
    }

    selectedLogId = logId;
    renderLogList(); // Re-render to show active state

    // Show analysis panel with loading state
    emptyState.style.display = 'none';
    analysisResults.style.display = 'block';
    loadingState.style.display = 'flex';
    resultsContent.style.display = 'none';

    // Update header info
    document.getElementById('analysis-job-name').textContent = log.name;
    document.getElementById('analysis-job-info').textContent = `${log.status} | ${log.timestamp}`;

    // Call real backend API
    try {
        const analysis = await analyzeLogWithBackend(log.id, selectedLLMId);
        
        if (analysis.success) {
            currentAnalysis = analysis.analysis;
            currentAnalysis.llmUsed = selectedLLMId;
            
            // Show results
            loadingState.style.display = 'none';
            resultsContent.style.display = 'block';
            
            // Populate results
            populateResults(currentAnalysis);
        } else {
            throw new Error(analysis.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis failed:', error);
        loadingState.style.display = 'none';
        resultsContent.innerHTML = `
            <div style="color: var(--accent-red); padding: 20px; text-align: center;">
                <p style="font-weight: 600; margin-bottom: 8px;">Analysis Failed</p>
                <p style="font-size: 14px;">${error.message}</p>
                <button onclick="handleAnalyze('${logId}')" style="margin-top: 16px; padding: 8px 16px; background: var(--accent-blue); color: white; border: none; border-radius: 6px; cursor: pointer;">
                    Retry
                </button>
            </div>
        `;
        resultsContent.style.display = 'block';
    }
}

// Populate the results section with analysis data
function populateResults(analysis) {
    document.getElementById('error-summary').textContent = analysis.errorSummary;
    document.getElementById('root-cause').textContent = analysis.rootCause;
    
    // Technical Details - format nicely if it contains structured info
    const stackTraceContent = document.getElementById('stack-trace');
    const detailsText = analysis.stackTrace;
    
    // Check if the content looks structured (has dashes or colons)
    if (detailsText.includes(':') || detailsText.includes('-')) {
        stackTraceContent.innerHTML = `<pre>${detailsText}</pre>`;
    } else {
        stackTraceContent.textContent = detailsText;
    }
    
    // Suggested actions as list
    const actionsList = document.getElementById('suggested-actions');
    actionsList.innerHTML = '';
    analysis.suggestedActions.forEach(action => {
        const li = document.createElement('li');
        li.textContent = action;
        actionsList.appendChild(li);
    });

    // Reset feedback buttons
    feedbackHelpful.classList.remove('active');
    feedbackNotHelpful.classList.remove('active');
}

// Close analysis panel
function closeAnalysis() {
    selectedLogId = null;
    currentAnalysis = null;
    renderLogList();
    
    analysisResults.style.display = 'none';
    emptyState.style.display = 'flex';
}

// Handle filter change
function handleFilterChange() {
    currentFilter = filterSelector.value;
    fetchJobs(); // Fetch new jobs with updated filter
}

// Handle stack trace expand/collapse
function toggleStackTrace(event) {
    console.log('toggleStackTrace called');
    
    // Prevent event bubbling
    if (event) {
        event.stopPropagation();
    }
    
    // Check if elements exist
    if (!stackTraceContent || !expandStackTraceBtn) {
        console.error('Stack trace elements not found!', {
            stackTraceContent,
            expandStackTraceBtn
        });
        return;
    }
    
    const isExpanded = stackTraceContent.style.display === 'block';
    const expandText = expandStackTraceBtn.querySelector('.expand-text');
    
    console.log('Current state:', {
        isExpanded,
        display: stackTraceContent.style.display,
        expandText: expandText ? expandText.textContent : 'not found'
    });
    
    if (isExpanded) {
        stackTraceContent.style.display = 'none';
        expandStackTraceBtn.classList.remove('expanded');
        if (expandText) {
            expandText.textContent = 'Show Context';
        }
        console.log('Collapsed technical details');
    } else {
        stackTraceContent.style.display = 'block';
        expandStackTraceBtn.classList.add('expanded');
        if (expandText) {
            expandText.textContent = 'Hide Context';
        }
        console.log('Expanded technical details');
    }
}

// Handle copy button clicks
function handleCopy(button) {
    const targetId = button.dataset.copyTarget;
    const content = document.getElementById(targetId).textContent;
    
    navigator.clipboard.writeText(content).then(() => {
        // Show copied state
        button.classList.add('copied');
        const originalHTML = button.innerHTML;
        button.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Copied!
        `;
        
        // Reset after 2 seconds
        setTimeout(() => {
            button.classList.remove('copied');
            button.innerHTML = originalHTML;
        }, 2000);
    });
}

// Handle feedback buttons
function handleFeedback(isHelpful) {
    if (!currentAnalysis) return;

    // Toggle active state
    if (isHelpful) {
        feedbackHelpful.classList.toggle('active');
        feedbackNotHelpful.classList.remove('active');
    } else {
        feedbackNotHelpful.classList.toggle('active');
        feedbackHelpful.classList.remove('active');
    }

    // TODO: Send feedback to backend
    console.log('Feedback submitted:', {
        logId: selectedLogId,
        helpful: isHelpful,
        llm: currentAnalysis.llmUsed
    });
}

// Attach event listeners
function attachEventListeners() {
    console.log('Attaching event listeners...');
    
    // Filter selector
    filterSelector.addEventListener('change', handleFilterChange);

    // LLM connection selector - load models when connection changes
    llmConnectionSelector.addEventListener('change', (e) => {
        const connectionId = e.target.value;
        if (connectionId) {
            fetchLLMModels(connectionId);
        } else {
            llmModelSelector.innerHTML = '<option value="">Select connection first</option>';
            llmModelSelector.disabled = true;
        }
    });

    // Close analysis button
    closeAnalysisBtn.addEventListener('click', closeAnalysis);

    // Stack trace expand/collapse
    console.log('Attaching listeners to expand button:', expandStackTraceBtn);
    console.log('Stack trace content element:', stackTraceContent);
    
    if (expandStackTraceBtn) {
        expandStackTraceBtn.addEventListener('click', toggleStackTrace);
        console.log('Listener attached to expand button');
    } else {
        console.error('expandStackTraceBtn not found!');
    }
    
    const stackTraceHeader = document.getElementById('stack-trace-header');
    if (stackTraceHeader) {
        stackTraceHeader.addEventListener('click', toggleStackTrace);
        console.log('Listener attached to header');
    } else {
        console.error('stack-trace-header not found!');
    }

    // Copy buttons
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', () => handleCopy(button));
    });

    // Feedback buttons
    feedbackHelpful.addEventListener('click', () => handleFeedback(true));
    feedbackNotHelpful.addEventListener('click', () => handleFeedback(false));
    
    console.log('All event listeners attached');
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// ===========================
// Backend Integration Functions
// ===========================

/**
 * PHASE 1: Currently fetching real Dataiku jobs ✓
 * Using real backend API at /api/jobs
 */

/**
 * PHASE 2: To be implemented
 * This will integrate:
 * 1. LogAI preprocessing to extract structured error data
 * 2. LLM Mesh integration for AI-powered analysis
 * 3. Real-time analysis results
 */

/**
 * Analyze log with AI using real backend (Phase 2 implementation)
 * Calls backend /api/analyze endpoint with LLM Mesh
 */
async function analyzeLogWithBackend(logId, llmId) {
    const url = getWebAppBackendUrl('api/analyze');
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            job_id: logId,
            llm_id: llmId
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Analysis request failed');
    }
    
    const data = await response.json();
    return data;
}

/**
 * Submit feedback to backend
 */
async function submitFeedback(logId, isHelpful, llmUsed) {
    // TODO: Implement feedback endpoint in backend
    console.log('Feedback submitted:', { logId, isHelpful, llmUsed });
}