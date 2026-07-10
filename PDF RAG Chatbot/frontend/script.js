const uploadButton = document.getElementById('uploadButton');
const clearFileButton = document.getElementById('clearFileButton');
const clearChatButton = document.getElementById('clearChatButton');
const askButton = document.getElementById('askButton');
const questionInput = document.getElementById('questionInput');
const pdfInput = document.getElementById('pdfInput');
const dropzone = document.getElementById('dropzone');
const chatWindow = document.getElementById('chatWindow');
const uploadMessage = document.getElementById('uploadMessage');
const uploadStatus = document.getElementById('uploadStatus');
const uploadedFileName = document.getElementById('uploadedFileName');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = uploadProgress.querySelector('.progress-fill');

let selectedFile = null;
let isAsking = false;

function setUploadStatus(message, type = 'info') {
  uploadStatus.textContent = message;
  uploadStatus.style.color = type === 'error' ? '#ff6b6b' : type === 'success' ? '#3ddc97' : '#5f8cff';
}

function showMessage(element, message, type) {
  element.hidden = false;
  element.className = `message-box ${type}`;
  element.textContent = message;
}

function hideMessage(element) {
  element.hidden = true;
}

function resetUploadUI() {
  uploadProgress.hidden = true;
  progressFill.style.width = '0%';
  uploadMessage.hidden = true;
  uploadedFileName.textContent = '';
  selectedFile = null;
  pdfInput.value = '';
}

function addMessage(role, content, metadata = null) {
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = `
    <div class="bubble-title">${role === 'user' ? 'You' : 'Assistant'}</div>
    <div>${content}</div>
  `;

  if (metadata) {
    const meta = document.createElement('div');
    meta.className = 'bubble-meta';
    meta.textContent = metadata;
    bubble.appendChild(meta);
  }

  if (role === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'bubble-actions';
    const copyButton = document.createElement('button');
    copyButton.textContent = 'Copy';
    copyButton.addEventListener('click', () => {
      navigator.clipboard.writeText(content);
      copyButton.textContent = 'Copied';
      setTimeout(() => (copyButton.textContent = 'Copy'), 1200);
    });
    actions.appendChild(copyButton);
    bubble.appendChild(actions);
  }

  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showLoadingBubble() {
  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant';
  bubble.innerHTML = `
    <div class="bubble-title">Assistant</div>
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>
  `;
  bubble.id = 'loadingBubble';
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function removeLoadingBubble() {
  const bubble = document.getElementById('loadingBubble');
  if (bubble) {
    bubble.remove();
  }
}

function handleFileSelection(file) {
  if (!file) return;
  if (file.type !== 'application/pdf') {
    showMessage(uploadMessage, 'Please select a valid PDF file.', 'error');
    setUploadStatus('Invalid file', 'error');
    return;
  }
  selectedFile = file;
  uploadedFileName.textContent = `Selected: ${file.name}`;
  setUploadStatus('PDF ready to upload', 'success');
  hideMessage(uploadMessage);
}

pdfInput.addEventListener('change', (event) => {
  handleFileSelection(event.target.files[0]);
});

dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragover');
  handleFileSelection(event.dataTransfer.files[0]);
});

uploadButton.addEventListener('click', async () => {
  if (!selectedFile) {
    showMessage(uploadMessage, 'Please choose a PDF before uploading.', 'error');
    setUploadStatus('No file selected', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('upload_file', selectedFile);

  uploadProgress.hidden = false;
  uploadButton.disabled = true;
  clearFileButton.disabled = true;
  setUploadStatus('Uploading...', 'info');
  hideMessage(uploadMessage);

  try {
    const response = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.detail || 'Upload failed.');
    }

    showMessage(uploadMessage, `Uploaded successfully: ${result.filename || selectedFile.name}`, 'success');
    setUploadStatus('Upload complete', 'success');
    uploadedFileName.textContent = `Uploaded: ${result.filename || selectedFile.name}`;
    progressFill.style.width = '100%';
  } catch (error) {
    showMessage(uploadMessage, error.message || 'Upload failed.', 'error');
    setUploadStatus('Upload failed', 'error');
  } finally {
    uploadButton.disabled = false;
    clearFileButton.disabled = false;
  }
});

clearFileButton.addEventListener('click', () => {
  resetUploadUI();
  setUploadStatus('Ready', 'info');
});

clearChatButton.addEventListener('click', () => {
  chatWindow.innerHTML = '';
});

askButton.addEventListener('click', sendQuestion);

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendQuestion();
  }
});

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    showMessage(uploadMessage, 'Please enter a question.', 'error');
    return;
  }

  if (!selectedFile) {
    showMessage(uploadMessage, 'Please upload a PDF first.', 'error');
    return;
  }

  if (isAsking) return;
  isAsking = true;
  askButton.disabled = true;
  questionInput.disabled = true;
  addMessage('user', question);
  showLoadingBubble();
  hideMessage(uploadMessage);

  try {
    const response = await fetch('http://127.0.0.1:8000/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: 3 }),
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.detail || 'Could not generate an answer.');
    }

    removeLoadingBubble();
    const answerText = result.answer || 'No answer returned.';
    const sourceText = Array.isArray(result.sources) && result.sources.length
      ? result.sources.map((source) => `• Chunk ${source.chunk} · score ${source.score}`).join('<br>')
      : 'No sources returned.';

    addMessage('assistant', answerText, 'Generated from the uploaded document');
    const sourcesBubble = document.createElement('div');
    sourcesBubble.className = 'bubble assistant';
    sourcesBubble.innerHTML = `
      <div class="bubble-title">Sources</div>
      <div>${sourceText}</div>
    `;
    chatWindow.appendChild(sourcesBubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  } catch (error) {
    removeLoadingBubble();
    addMessage('assistant', `Error: ${error.message || 'Unable to process your request.'}`);
  } finally {
    isAsking = false;
    askButton.disabled = false;
    questionInput.disabled = false;
    questionInput.value = '';
    questionInput.focus();
  }
}
