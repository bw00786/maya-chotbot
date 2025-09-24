import React, { useState, useRef, useEffect, createContext, useContext } from 'react';
import axios from 'axios';
import {
  Container,
  Box,
  TextField,
  IconButton,
  CircularProgress,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  ThemeProvider,
  createTheme,
  CssBaseline,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

// Create a new context for the chat state
const ChatContext = createContext();

// Custom hook to use the chat context
function useChat() {
  return useContext(ChatContext);
}

// Custom hook for speech recognition
function useSpeech() {
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error("Speech Recognition not supported in this browser.");
      return;
    }

    const rec = new SpeechRecognition();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onend = () => setListening(false);

    recognitionRef.current = rec;

    // Clean up function
    return () => {
      if (rec) {
        rec.onend = null;
      }
    };
  }, []);

  function start(onResult) {
    const rec = recognitionRef.current;
    if (!rec) return;

    rec.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onResult(transcript);
    };
    rec.start();
    setListening(true);
  }

  function stop() {
    const rec = recognitionRef.current;
    if (!rec) return;
    rec.stop();
    setListening(false);
  }

  return { listening, start, stop };
}

// The Chat Provider component
function ChatProvider({ children }) {
  const [messages, setMessages] = useState([
    { role: 'system', content: 'You are a friendly conversational assistant named Mira. Keep replies short and warm.' }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ragMode, setRagMode] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  // State for Text-to-Speech voices
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('');

  // The base URL for the API
  const apiBase = 'http://localhost:8100';

  // Function to reset the chat
  const resetChat = () => {
    setMessages([{ role: 'system', content: 'You are a friendly conversational assistant named Mira. Keep replies short and warm.' }]);
    setError(null);
  };

  // Effect to load available voices for Text-to-Speech
  useEffect(() => {
    const fetchVoices = () => {
      const availableVoices = window.speechSynthesis.getVoices();
      setVoices(availableVoices);
      // Set a default voice if available
      if (availableVoices.length > 0) {
        setSelectedVoice(availableVoices[0].name);
      }
    };

    // Listen for voices being loaded
    window.speechSynthesis.addEventListener('voiceschanged', fetchVoices);
    // Initial fetch in case voices are already loaded
    fetchVoices();

    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', fetchVoices);
    };
  }, []);

  // Function to handle Text-to-Speech
  const speakMessage = (text, voiceName) => {
    const utterance = new SpeechSynthesisUtterance(text);
    const selected = voices.find(v => v.name === voiceName);
    if (selected) {
      utterance.voice = selected;
    } else {
      console.warn(`Voice "${voiceName}" not found. Using default.`);
    }
    window.speechSynthesis.speak(utterance);
  };

  // Function to handle sending a message
  async function sendMessage(text) {
    if (!text.trim()) return;

    const newMessage = { role: 'user', content: text };
    let updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setLoading(true);
    setError(null);

    try {
      let response;
      if (ragMode) {
        // Use the RAG endpoint and send the query
        response = await axios.post(`${apiBase}/rag_chat`, { query: text });
      } else {
        // Use the standard chat endpoint and send the full message history
        response = await axios.post(`${apiBase}/chat`, { messages: updatedMessages });
      }

      const assistantReply = response.data.reply;
      const finalMessages = [...updatedMessages, { role: 'assistant', content: assistantReply }];
      setMessages(finalMessages);

      // Speak the assistant's reply
      speakMessage(assistantReply, selectedVoice);

    } catch (err) {
      console.error('API Error:', err);
      setError('Failed to get a response from the assistant. Please try again.');
      setMessages(messages); // Revert messages on error
    } finally {
      setLoading(false);
    }
  }

  const contextValue = {
    messages,
    loading,
    error,
    ragMode,
    setRagMode,
    voices,
    selectedVoice,
    setSelectedVoice,
    sendMessage,
    darkMode,
    setDarkMode,
    resetChat,
  };

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
}

// New component to contain the UI and consume the context
function ChatUI() {
  const { messages, loading, error, ragMode, setRagMode, voices, selectedVoice, setSelectedVoice, sendMessage, darkMode, setDarkMode, resetChat } = useChat();
  const [input, setInput] = useState('');
  const { listening, start, stop } = useSpeech();

  // Function to handle voice input
  const handleVoiceInput = () => {
    if (listening) {
      stop();
    } else {
      start((transcript) => {
        setInput(transcript);
        sendMessage(transcript);
      });
    }
  };

  // JSX for the UI
  return (
    <Container maxWidth="sm" sx={{ display: 'flex', flexDirection: 'column', height: '100vh', py: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1" align="center">
          Mira Assistant
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton onClick={resetChat} color="primary" aria-label="new chat">
            <DeleteOutlineIcon />
          </IconButton>
          <FormControlLabel
            control={<Switch checked={darkMode} onChange={() => setDarkMode(!darkMode)} />}
            label="Dark Mode"
          />
        </Box>
      </Box>

      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 1, display: 'flex', flexDirection: 'column', mb: 2 }}>
        {messages.filter(m => m.role !== 'system').map((m, i) => (
          <Paper key={i} sx={{
            p: 1,
            mb: 1,
            maxWidth: '80%',
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            bgcolor: m.role === 'user' ? (darkMode ? '#37474f' : '#e3f2fd') : (darkMode ? '#455a64' : '#f5f5f5'),
            borderRadius: '10px'
          }}>
            <Typography variant="caption" color="text.secondary">{m.role}</Typography>
            <Typography>{m.content}</Typography>
          </Paper>
        ))}
        {loading && <CircularProgress size={24} sx={{ alignSelf: 'center', my: 1 }} />}
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <FormControlLabel
            control={<Switch checked={ragMode} onChange={() => setRagMode(!ragMode)} />}
            label="RAG Mode"
          />
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <TextField
            fullWidth
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') { sendMessage(input); setInput(''); } }}
            placeholder="Type a message..."
            disabled={loading || listening}
          />
          <IconButton color="primary" onClick={() => { sendMessage(input); setInput(''); }} disabled={loading || listening}>
            <SendIcon />
          </IconButton>
          <IconButton color={listening ? "error" : "primary"} onClick={handleVoiceInput}>
            {listening ? <StopIcon /> : <MicIcon />}
          </IconButton>
        </Box>
      </Box>

      <FormControl fullWidth sx={{ mt: 2 }}>
        <InputLabel>Voice</InputLabel>
        <Select
          value={selectedVoice}
          label="Voice"
          onChange={e => setSelectedVoice(e.target.value)}
        >
          {voices.map((v, i) => (
            <MenuItem key={i} value={v.name}>{v.name} ({v.lang})</MenuItem>
          ))}
        </Select>
      </FormControl>

      {error && <Typography color="error" sx={{ mt: 1 }}>{error}</Typography>}
    </Container>
  );
}

// The main App component which acts as the provider wrapper
export default function App() {
  const [newChatKey, setNewChatKey] = useState(0);

  const theme = createTheme({
    palette: {
      mode: 'light',
    },
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ChatProvider>
        <ChatUI key={newChatKey} />
      </ChatProvider>
    </ThemeProvider>
  );
}