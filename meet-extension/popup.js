// popup.js
let isEnabled = false;

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('sourceImage');
    const file = fileInput.files[0];
    
    if (!file) {
        updateStatus('Please select a file first');
        return;
    }

    updateStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/upload_source', {
            method: 'POST',
            body: formData,
        });
        
        const result = await response.json();
        if (result.face_id) {
            await chrome.storage.local.set({ face_id: result.face_id });
            updateStatus('Upload successful! You can now start face swap.');
            document.getElementById('toggleSwap').disabled = false;
        } else {
            updateStatus('Upload failed: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        updateStatus('Upload failed: ' + error.message);
        console.error('Upload error:', error);
    }
});

document.getElementById('toggleSwap').addEventListener('click', async () => {
    isEnabled = !isEnabled;
    const button = document.getElementById('toggleSwap');
    button.textContent = isEnabled ? 'Stop Face Swap' : 'Start Face Swap';
    
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { 
            type: 'TOGGLE_SWAP',
            enabled: isEnabled 
        });
        updateStatus(isEnabled ? 'Face swap running...' : 'Face swap stopped');
    }
});

// Check initial state
document.addEventListener('DOMContentLoaded', async () => {
    const button = document.getElementById('toggleSwap');
    button.disabled = true;
    
    try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
            updateStatus('Ready to use. Please upload a source face.');
        } else {
            updateStatus('Server is not responding. Please check if it\'s running.');
        }
    } catch (error) {
        updateStatus('Cannot connect to server. Is it running on localhost:8000?');
    }
});