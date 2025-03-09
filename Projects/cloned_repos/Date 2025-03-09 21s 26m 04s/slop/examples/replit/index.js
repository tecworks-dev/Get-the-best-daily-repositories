
const express = require('express');
const app = express();
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Trivia game data
const triviaQuestions = [
  {
    question: "What is the capital of France?",
    answer: "paris",
    hint: "It's known as the City of Light."
  },
  {
    question: "Which planet is known as the Red Planet?",
    answer: "mars",
    hint: "It's named after the Roman god of war."
  },
  {
    question: "What is the largest mammal in the world?",
    answer: "blue whale",
    hint: "It lives in the ocean and can weigh up to 200 tons."
  },
  {
    question: "Who painted the Mona Lisa?",
    answer: "leonardo da vinci",
    hint: "He was an Italian polymath from the Renaissance period."
  },
  {
    question: "What is the chemical symbol for gold?",
    answer: "au",
    hint: "It comes from the Latin word 'aurum'."
  }
];

// Game state
let currentGame = {
  active: false,
  currentQuestion: null,
  score: 0,
  askedQuestions: [],
  hintUsed: false
};

// In-memory storage for SLOP
const memory = new Map();

// 1. CHAT ENDPOINT - SLOP compatible
app.post('/chat', (req, res) => {
  const message = req.body.messages?.[0]?.content || '';
  const lowerMessage = message.toLowerCase();
  
  // Response based on game state and message
  let response = '';

  // Trivia game commands
  if (lowerMessage.includes('start trivia') || lowerMessage.includes('play trivia')) {
    currentGame = {
      active: true,
      currentQuestion: null,
      score: 0,
      askedQuestions: [],
      hintUsed: false
    };
    response = "Welcome to Trivia Challenge! I'll ask you questions and you try to answer them. Say 'next question' to begin!";
  } 
  else if (currentGame.active && (lowerMessage.includes('next question') || lowerMessage.includes('new question'))) {
    // Get a question that hasn't been asked yet
    const availableQuestions = triviaQuestions.filter(q => !currentGame.askedQuestions.includes(q.question));
    
    if (availableQuestions.length === 0) {
      response = `Game over! Your final score is ${currentGame.score}/${triviaQuestions.length}. Say 'start trivia' to play again!`;
      currentGame.active = false;
    } else {
      currentGame.currentQuestion = availableQuestions[Math.floor(Math.random() * availableQuestions.length)];
      currentGame.askedQuestions.push(currentGame.currentQuestion.question);
      currentGame.hintUsed = false;
      response = `Question: ${currentGame.currentQuestion.question} (Say 'hint' if you need help)`;
    }
  }
  else if (currentGame.active && lowerMessage.includes('hint') && currentGame.currentQuestion) {
    currentGame.hintUsed = true;
    response = `Hint: ${currentGame.currentQuestion.hint}`;
  }
  else if (currentGame.active && currentGame.currentQuestion && lowerMessage.includes('skip')) {
    response = `The answer was: ${currentGame.currentQuestion.answer}. Say 'next question' for another one!`;
    currentGame.currentQuestion = null;
  }
  else if (currentGame.active && currentGame.currentQuestion) {
    // Check if the answer is correct
    if (lowerMessage.includes(currentGame.currentQuestion.answer.toLowerCase())) {
      currentGame.score += currentGame.hintUsed ? 0.5 : 1; // Half point if hint was used
      response = currentGame.hintUsed ? 
        `Correct! You get half a point for using a hint. Your score is now ${currentGame.score}. Say 'next question' to continue!` :
        `Correct! Your score is now ${currentGame.score}. Say 'next question' to continue!`;
      currentGame.currentQuestion = null;
    } else {
      response = "Sorry, that's not correct. Try again, say 'hint' for a clue, or 'skip' to move on.";
    }
  }
  else if (lowerMessage.includes('stop trivia') || lowerMessage.includes('end trivia')) {
    response = `Game ended. Your final score was ${currentGame.score}. Thanks for playing!`;
    currentGame.active = false;
  }
  // Standard responses if not in game mode
  else if (lowerMessage.includes('hello')) {
    response = "Hello there! Want to play a trivia game? Say 'start trivia' to begin!";
  } else if (lowerMessage.includes('weather')) {
    response = "I don't have real-time weather data, but I hope it's sunny where you are!";
  } else if (lowerMessage.includes('name')) {
    response = "I'm a Trivia Bot. Nice to meet you! Say 'start trivia' to play a game.";
  } else {
    response = `You said: "${message}". Try saying 'start trivia' to play a fun trivia game!`;
  }

  res.json({ message: { role: 'assistant', content: response } });
});

// 2. TOOLS ENDPOINT - SLOP compatible
app.get('/tools', (req, res) => {
  res.json({ 
    tools: [
      { 
        id: 'trivia', 
        description: 'Play a trivia game with questions on various subjects' 
      },
      { 
        id: 'hint', 
        description: 'Get a hint for the current question in the trivia game' 
      },
      { 
        id: 'score', 
        description: 'Check your current score in the trivia game' 
      }
    ] 
  });
});

// Tool execution endpoint
app.post('/tools/:tool_id', (req, res) => {
  const toolId = req.params.tool_id;
  
  // Ensure req.body is initialized even if no JSON body is sent
  req.body = req.body || {};
  
  switch(toolId) {
    case 'trivia':
      if (!currentGame.active) {
        currentGame = {
          active: true,
          currentQuestion: null,
          score: 0,
          askedQuestions: [],
          hintUsed: false
        };
        res.json({ result: "Trivia game started! Say 'next question' to begin." });
      } else {
        res.json({ result: "You're already in a trivia game! Say 'next question' for a new question or 'end trivia' to stop." });
      }
      break;
      
    case 'hint':
      if (currentGame.active && currentGame.currentQuestion) {
        currentGame.hintUsed = true;
        res.json({ result: `Hint: ${currentGame.currentQuestion.hint}` });
      } else {
        res.json({ result: "No active question to give a hint for. Start a game with 'start trivia' first!" });
      }
      break;
      
    case 'score':
      if (currentGame.active) {
        res.json({ 
          result: `Your current score is ${currentGame.score}. You've answered ${currentGame.askedQuestions.length} questions.` 
        });
      } else {
        res.json({ result: "No active game. Start a new game with 'start trivia'!" });
      }
      break;
      
    default:
      res.status(404).json({ error: "Tool not found" });
  }
});

// 3. MEMORY ENDPOINT - SLOP compatible
app.post('/memory', (req, res) => {
  const { key, value } = req.body;
  if (key && value !== undefined) {
    memory.set(key, value);
    res.json({ status: 'stored' });
  } else {
    res.status(400).json({ error: 'Both key and value are required' });
  }
});

app.get('/memory/:key', (req, res) => {
  const { key } = req.params;
  if (memory.has(key)) {
    res.json({ value: memory.get(key) });
  } else {
    res.status(404).json({ error: 'Key not found' });
  }
});

app.get('/memory', (req, res) => {
  const keys = Array.from(memory.keys()).map(key => ({
    key,
    created_at: new Date().toISOString()
  }));
  res.json({ keys });
});

// 4. RESOURCES ENDPOINT - SLOP compatible
app.get('/resources', (req, res) => {
  res.json({ 
    resources: [
      { 
        id: 'trivia-questions', 
        title: 'Available Trivia Questions',
        type: 'collection' 
      },
      { 
        id: 'commands', 
        title: 'Trivia Game Commands',
        type: 'guide' 
      }
    ] 
  });
});

app.get('/resources/:id', (req, res) => {
  const resourceId = req.params.id;
  
  switch (resourceId) {
    case 'trivia-questions':
      // Return number of available questions and categories
      res.json({
        id: 'trivia-questions',
        title: 'Available Trivia Questions',
        content: `There are ${triviaQuestions.length} questions available covering topics like geography, science, art, and more.`,
        metadata: {
          count: triviaQuestions.length,
          last_updated: new Date().toISOString()
        }
      });
      break;
      
    case 'commands':
      res.json({
        id: 'commands',
        title: 'Trivia Game Commands',
        content: "Available commands: 'start trivia', 'next question', 'hint', 'skip', 'end trivia'",
        metadata: {
          command_count: 5,
          last_updated: new Date().toISOString()
        }
      });
      break;
      
    default:
      res.status(404).json({ error: 'Resource not found' });
  }
});

// Simple search for resources
app.get('/resources/search', (req, res) => {
  const query = req.query.q?.toLowerCase() || '';
  
  const results = [];
  
  if (query.includes('trivia') || query.includes('question')) {
    results.push({
      id: 'trivia-questions',
      title: 'Available Trivia Questions',
      type: 'collection',
      score: 0.95
    });
  }
  
  if (query.includes('command') || query.includes('help')) {
    results.push({
      id: 'commands',
      title: 'Trivia Game Commands',
      type: 'guide',
      score: 0.90
    });
  }
  
  res.json({ results });
});

// 5. PAY ENDPOINT - SLOP compatible (mock implementation)
app.post('/pay', (req, res) => {
  // Simple mock implementation
  const transactionId = `tx_${Date.now()}`;
  
  // Store transaction in memory
  memory.set(transactionId, {
    amount: req.body.amount || 0,
    currency: req.body.currency || 'USD',
    description: req.body.description || 'Trivia game usage',
    status: 'success',
    created_at: new Date().toISOString()
  });
  
  res.json({
    transaction_id: transactionId,
    status: 'success',
    receipt_url: `https://api.example.com/receipts/${transactionId}`
  });
});

app.get('/pay/:id', (req, res) => {
  const { id } = req.params;
  
  if (memory.has(id)) {
    const transaction = memory.get(id);
    res.json({
      transaction_id: id,
      ...transaction
    });
  } else {
    res.status(404).json({ error: 'Transaction not found' });
  }
});

// Start the server
app.listen(3000, '0.0.0.0', () => console.log('✨ SLOP running on port 3000'));
