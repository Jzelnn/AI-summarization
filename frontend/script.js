// Updated JavaScript for NLP Summarizer with Metrics and Multilingual Support

const API_BASE = "http://127.0.0.1:8000";

let currentData = {
  original: "",
  summary: "",
  entities: {},
  metrics: {},
  language: "en"
};

// Theme toggle
function themeToggle() {
  document.body.classList.toggle('dark-theme');
  const themeBtn = document.getElementById('themeToggle');
  themeBtn.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
}

// Tab switching
function switchTab(tab) {
  const textArea = document.getElementById('inputTextMessage');
  const fileArea = document.getElementById('inputFileMessage');
  const textBtn = document.getElementById('tabTextBtn');
  const fileBtn = document.getElementById('tabFileBtn');

  if (tab === 'text') {
    textArea.classList.remove('hidden');
    fileArea.classList.add('hidden');
    textBtn.classList.add('active');
    fileBtn.classList.remove('active');
  } else {
    textArea.classList.add('hidden');
    fileArea.classList.remove('hidden');
    textBtn.classList.remove('active');
    fileBtn.classList.add('active');
  }
}

// Voice input
function startDictation() {
  if (!('webkitSpeechRecognition' in window)) {
    alert('Speech recognition not supported in this browser.');
    return;
  }

  const recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  const micBtn = document.getElementById('micBtn');
  micBtn.textContent = '🎙️';
  micBtn.disabled = true;

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('textInputArea').value = transcript;
    micBtn.textContent = '🎤';
    micBtn.disabled = false;
  };

  recognition.onerror = () => {
    micBtn.textContent = '🎤';
    micBtn.disabled = false;
    alert('Speech recognition error');
  };

  recognition.start();
}

// Display metrics with visual bars
function displayMetrics(metrics) {
  if (!metrics || Object.keys(metrics).length === 0) {
    document.getElementById('metricsSection').classList.add('hidden');
    return;
  }

  document.getElementById('metricsSection').classList.remove('hidden');

  // Helper function to update metric bar
  function updateMetricBar(id, value) {
    const bar = document.getElementById(id + 'Bar');
    const valueSpan = document.getElementById(id + 'Value');
    
    if (bar && valueSpan && value !== undefined) {
      const percentage = (value * 100);
      bar.style.width = percentage + '%';
      
      // Color code based on value
      if (percentage >= 70) {
        bar.style.backgroundColor = '#10b981'; // green
      } else if (percentage >= 40) {
        bar.style.backgroundColor = '#f59e0b'; // orange
      } else {
        bar.style.backgroundColor = '#ef4444'; // red
      }
      
      valueSpan.textContent = (value * 100).toFixed(1) + '%';
    }
  }

  // Update ROUGE scores
  updateMetricBar('rouge1', metrics.rouge1_f1 || 0);
  updateMetricBar('rouge2', metrics.rouge2_f1 || 0);
  updateMetricBar('rougeL', metrics.rougeL_f1 || 0);

  // Update quality metrics
  updateMetricBar('bleu', metrics.bleu || 0);
  updateMetricBar('contentF1', metrics.content_f1 || 0);
  updateMetricBar('entity', metrics.entity_preservation || 0);

  // Update compression info
  document.getElementById('compressionValue').textContent = 
    ((metrics.compression_ratio || 0) * 100).toFixed(1) + '%';
  document.getElementById('originalWords').textContent = 
    metrics.original_words || '-';
  document.getElementById('summaryWords').textContent = 
    metrics.summary_words || '-';
}

// Display language info
function displayLanguageInfo(detectedLang, outputLang) {
  if (detectedLang || outputLang) {
    document.getElementById('languageSection').classList.remove('hidden');
    document.getElementById('detectedLang').textContent = detectedLang || 'Unknown';
    document.getElementById('outputLang').textContent = outputLang || detectedLang || 'Unknown';
  }
}

// Main generation function
async function generate() {
  const textInput = document.getElementById('textInputArea').value.trim();
  const fileInput = document.getElementById('fileInput').files[0];
  const aiModel = document.getElementById('aiModel').value;
  const summaryLevel = document.getElementById('summaryLevel').value;
  const outputLanguage = document.getElementById('outputLanguage').value;

  // Validation
  if (!textInput && !fileInput) {
    alert('Please provide text or upload a file');
    return;
  }

  // Show loading
  document.getElementById('loadingText').classList.remove('hidden');
  document.getElementById('generateBtn').disabled = true;

  try {
    let response;

    if (aiModel === 'compare') {
      // Model comparison mode
      await compareModels(textInput, summaryLevel);
      return;
    }

    // Build form data
    const formData = new FormData();
    formData.append('level', summaryLevel);
    formData.append('model', aiModel);
    if (outputLanguage) {
      formData.append('output_lang', outputLanguage);
    }

    // Determine endpoint
    if (fileInput) {
      formData.append('file', fileInput);
      response = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      });
    } else {
      formData.append('text', textInput);
      response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });
    }

    const data = await response.json();

    if (data.status === 'success') {
      // Store data
      currentData = {
        original: data.original_text || data.transcribed_text || textInput,
        summary: data.summary,
        entities: data.entities,
        metrics: data.evaluation_metrics || {},
        topic: data.topic,
        language: data.detected_language
      };

      // Display results
      displaySummary(currentData);
      displayLanguageInfo(data.detected_language, data.output_language);
      displayMetrics(data.evaluation_metrics);
      displayEntities(data.entities);
      
      // Show processing time
      document.getElementById('processingTime').textContent = data.processing_time;

      // Show download section
      document.getElementById('downloadSection').classList.remove('hidden');

      // Add to history
      addToHistory(currentData);
    } else {
      alert('Error: ' + (data.detail || 'Unknown error'));
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Failed to generate summary: ' + error.message);
  } finally {
    document.getElementById('loadingText').classList.add('hidden');
    document.getElementById('generateBtn').disabled = false;
  }
}

// Compare both models
async function compareModels(text, level) {
  try {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('level', level);

    const response = await fetch(`${API_BASE}/api/compare`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.status === 'success') {
      displayComparison(data);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Comparison failed: ' + error.message);
  } finally {
    document.getElementById('loadingText').classList.add('hidden');
    document.getElementById('generateBtn').disabled = false;
  }
}

// Display comparison results
function displayComparison(data) {
  // Hide single result section
  document.getElementById('resultSection').classList.add('hidden');
  document.getElementById('metricsSection').classList.add('hidden');
  
  // Show comparison section
  document.getElementById('comparisonSection').classList.remove('hidden');
  
  // Show topic
  const topicBadge = document.getElementById('topicBadgeCmp');
  if (data.topic) {
    topicBadge.textContent = `📌 Topic: ${data.topic}`;
    topicBadge.classList.remove('hidden');
  }

  // Display original text
  document.getElementById('cmpOriginalText').textContent = data.original_text;

  // Display Gemini results
  if (data.results.gemini) {
    const gemini = data.results.gemini;
    document.getElementById('geminiSummary').textContent = gemini.summary;
    document.getElementById('geminiTime').textContent = gemini.processing_time;
    document.getElementById('geminiWords').textContent = gemini.word_count;
    document.getElementById('geminiRatio').textContent = 
      ((gemini.metrics.compression_ratio || 0) * 100).toFixed(1) + '%';
    
    // Metrics
    document.getElementById('geminiRouge1').textContent = 
      ((gemini.metrics.rouge1_f1 || 0) * 100).toFixed(1) + '%';
    document.getElementById('geminiF1').textContent = 
      ((gemini.metrics.content_f1 || 0) * 100).toFixed(1) + '%';
    document.getElementById('geminiBleu').textContent = 
      ((gemini.metrics.bleu || 0) * 100).toFixed(1) + '%';
  }

  // Display HuggingFace results
  if (data.results.huggingface) {
    const hf = data.results.huggingface;
    document.getElementById('huggingfaceSummary').textContent = hf.summary;
    document.getElementById('hfTime').textContent = hf.processing_time;
    document.getElementById('hfWords').textContent = hf.word_count;
    document.getElementById('hfRatio').textContent = 
      ((hf.metrics.compression_ratio || 0) * 100).toFixed(1) + '%';
    
    // Metrics
    document.getElementById('hfRouge1').textContent = 
      ((hf.metrics.rouge1_f1 || 0) * 100).toFixed(1) + '%';
    document.getElementById('hfF1').textContent = 
      ((hf.metrics.content_f1 || 0) * 100).toFixed(1) + '%';
    document.getElementById('hfBleu').textContent = 
      ((hf.metrics.bleu || 0) * 100).toFixed(1) + '%';
  }

  // Show winner
  if (data.winner) {
    const winnerBadge = document.getElementById('winnerBadge');
    winnerBadge.classList.remove('hidden');
    document.getElementById('winnerModel').textContent = 
      data.winner === 'gemini' ? '🚀 Gemini 2.5 Flash' : '🤗 T5-Small';
  }

  // Show download section
  document.getElementById('downloadSection').classList.remove('hidden');
}

// Display single summary
function displaySummary(data) {
  document.getElementById('resultSection').classList.remove('hidden');
  document.getElementById('comparisonSection').classList.add('hidden');

  // Show topic
  const topicBadge = document.getElementById('topicBadge');
  if (data.topic) {
    topicBadge.textContent = `📌 Topic: ${data.topic}`;
    topicBadge.classList.remove('hidden');
  }

  document.getElementById('beforeText').textContent = data.original;
  document.getElementById('afterText').textContent = data.summary;
}

// Display entities
function displayEntities(entities) {
  const entityList = document.getElementById('entityList');
  entityList.innerHTML = '';

  if (!entities || Object.keys(entities).length === 0) {
    document.getElementById('entitySection').classList.add('hidden');
    return;
  }

  document.getElementById('entitySection').classList.remove('hidden');

  for (const [type, items] of Object.entries(entities)) {
    if (items.length > 0) {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${type}:</strong> ${items.join(', ')}`;
      entityList.appendChild(li);
    }
  }
}

// Copy functions
function copySummary() {
  navigator.clipboard.writeText(currentData.summary);
  showNotification('Summary copied!');
}

function copyEntities() {
  const text = Object.entries(currentData.entities)
    .map(([type, items]) => `${type}: ${items.join(', ')}`)
    .join('\n');
  navigator.clipboard.writeText(text);
  showNotification('Entities copied!');
}

function copyGeminiSummary() {
  const summary = document.getElementById('geminiSummary').textContent;
  navigator.clipboard.writeText(summary);
  showNotification('Gemini summary copied!');
}

function copyHuggingFaceSummary() {
  const summary = document.getElementById('huggingfaceSummary').textContent;
  navigator.clipboard.writeText(summary);
  showNotification('T5-Small summary copied!');
}

function copyAll() {
  const allData = `
ORIGINAL TEXT:
${currentData.original}

SUMMARY:
${currentData.summary}

ENTITIES:
${Object.entries(currentData.entities).map(([type, items]) => `${type}: ${items.join(', ')}`).join('\n')}

METRICS:
${Object.entries(currentData.metrics).map(([key, value]) => `${key}: ${typeof value === 'number' ? (value * 100).toFixed(1) + '%' : value}`).join('\n')}
  `.trim();
  
  navigator.clipboard.writeText(allData);
  showNotification('All data copied!');
}

// Download functions
function downloadTXT() {
  const content = `SUMMARY\n\n${currentData.summary}\n\nORIGINAL TEXT\n\n${currentData.original}`;
  const blob = new Blob([content], { type: 'text/plain' });
  downloadFile(blob, 'summary.txt');
}

function downloadDOC() {
  // Simple DOC format
  const content = `<html><body><h1>Summary</h1><p>${currentData.summary}</p><h1>Original</h1><p>${currentData.original}</p></body></html>`;
  const blob = new Blob([content], { type: 'application/msword' });
  downloadFile(blob, 'summary.doc');
}

function downloadPDF() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  
  doc.setFontSize(16);
  doc.text('Summary', 10, 10);
  
  doc.setFontSize(12);
  const summaryLines = doc.splitTextToSize(currentData.summary, 180);
  doc.text(summaryLines, 10, 20);
  
  doc.save('summary.pdf');
}

function downloadFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// History management
function addToHistory(data) {
  let history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
  
  history.unshift({
    timestamp: new Date().toISOString(),
    original: data.original.substring(0, 100) + '...',
    summary: data.summary.substring(0, 100) + '...',
    topic: data.topic
  });
  
  // Keep only last 10
  history = history.slice(0, 10);
  
  localStorage.setItem('summaryHistory', JSON.stringify(history));
  displayHistory();
}

function displayHistory() {
  const history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
  const historyList = document.getElementById('historyList');
  
  if (history.length === 0) {
    document.getElementById('historySection').classList.add('hidden');
    return;
  }
  
  document.getElementById('historySection').classList.remove('hidden');
  historyList.innerHTML = '';
  
  history.forEach((item, index) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <strong>${new Date(item.timestamp).toLocaleString()}</strong><br>
      Topic: ${item.topic || 'general'}<br>
      <small>${item.summary}</small>
    `;
    historyList.appendChild(li);
  });
}

// Clear all
function clearAll() {
  document.getElementById('textInputArea').value = '';
  document.getElementById('fileInput').value = '';
  
  document.getElementById('resultSection').classList.add('hidden');
  document.getElementById('comparisonSection').classList.add('hidden');
  document.getElementById('metricsSection').classList.add('hidden');
  document.getElementById('languageSection').classList.add('hidden');
  document.getElementById('entitySection').classList.add('hidden');
  document.getElementById('downloadSection').classList.add('hidden');
  
  currentData = { original: "", summary: "", entities: {}, metrics: {} };
}

// Notification
function showNotification(message) {
  const notification = document.createElement('div');
  notification.className = 'notification';
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.classList.add('show');
  }, 100);
  
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  }, 2000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  displayHistory();
  
  // File upload UI
  const fileInput = document.getElementById('fileInput');
  const uploadText = document.getElementById('uploadText');
  
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      uploadText.textContent = `📄 ${e.target.files[0].name}`;
    }
  });
});