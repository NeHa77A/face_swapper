chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'TOGGLE_SWAP') {
    chrome.tabs.sendMessage(sender.tab.id, { 
      type: 'TOGGLE_SWAP',
      enabled: message.enabled 
    });
  }
});