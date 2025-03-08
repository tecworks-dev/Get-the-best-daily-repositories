# SLOP Python Example

A simple implementation of the [SLOP](https://github.com/agnt-gg/slop) pattern in Python.

## Python Quick Start

```bash
# Clone the repo
git clone https://github.com/agnt-gg/slop
cd slop/python

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run it
python slop.py
```

## Endpoints

```python
# CHAT - Talk to AI
POST /chat
{
  "messages": [{ "content": "Hello SLOP!" }]
}

# TOOLS - Use tools
GET /tools
POST /tools/calculator { "expression": "2 + 2" }
POST /tools/greet { "name": "SLOP" }

# MEMORY - Store data
POST /memory { "key": "test", "value": "hello" }
GET /memory/test

# RESOURCES - Get knowledge
GET /resources
GET /resources/hello

# PAY - Handle payments
POST /pay { "amount": 10 }
```

## Structure

- `slop.py` - The entire implementation
- `requirements.txt` - Dependencies

That's it. Just two files.

## Dependencies

- `flask` - For clean routing
- `requests` - For testing endpoints

## Try It

After starting the server, it automatically runs tests for all endpoints:

```bash
python slop.py

# Output:
✨ SLOP running on http://localhost:5000
🚀 Running tests...

📝 Testing chat...
You said: Hello SLOP!

🔧 Testing tools...
2 + 2 = 4
Hello, SLOP!

💾 Testing memory...
Stored value: hello world

📚 Testing resources...
Resource content: Hello, SLOP!

💰 Testing pay...
Transaction: tx_1234567890

✅ All tests passed!
```

## Learn More

Check out the [main SLOP repository](https://github.com/agnt-gg/slop) for:
- Full specification
- Other language examples
- Core concepts
- Best practices

Remember: SLOP is just a pattern. This is a simple implementation example to show how it works.