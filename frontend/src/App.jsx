import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';

const API_BASE = '';  // Empty = relative URL, works in both dev (proxied) and production

function App() {
  const [provider, setProvider] = useState('Groq');
  const [apiKey, setApiKey] = useState('');
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useContextual, setUseContextual] = useState(false);
  const [useHyDE, setUseHyDE] = useState(false);

  const handleUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (!selectedFiles.length) return;
    setFiles(prev => [...prev, ...selectedFiles]);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      selectedFiles.forEach(f => formData.append('files', f));
      await axios.post(`${API_BASE}/ingest`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    } catch (err) {
      console.error("Ingestion failed:", err.response?.data?.detail || err.message);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ PDF ingestion failed: ${err.response?.data?.detail || 'Check API key and try again.'}`
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        query: input,
        api_key: apiKey || undefined,
        provider: provider,
        use_reranker: true,
        top_k: 5
      });

      const assistantMsg = {
        role: 'assistant',
        content: response.data.answer,
        citations: response.data.citations
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "❌ Error: Failed to get response from DocuGenie. Please check your API key and backend connection."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex w-full h-full bg-transparent text-white overflow-hidden font-sans selection:bg-brand-500/30">
      <Sidebar
        provider={provider}
        setProvider={setProvider}
        apiKey={apiKey}
        setApiKey={setApiKey}
        files={files}
        handleUpload={handleUpload}
        isProcessing={isProcessing}
        useContextual={useContextual}
        setUseContextual={setUseContextual}
        useHyDE={useHyDE}
        setUseHyDE={setUseHyDE}
      />
      <main className="flex-1 relative">
        <Chat
          messages={messages}
          input={input}
          setInput={setInput}
          handleSend={handleSend}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}

export default App;
