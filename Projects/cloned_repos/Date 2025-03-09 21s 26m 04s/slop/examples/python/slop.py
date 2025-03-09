# Python implementation of the SLOP pattern

from flask import Flask, request, jsonify
import requests
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Available tools and resources
tools = {
    'calculator': {
        'id': 'calculator',
        'description': 'Basic math',
        'execute': lambda params: {'result': eval(params['expression'])}
    },
    'greet': {
        'id': 'greet',
        'description': 'Says hello',
        'execute': lambda params: {'result': f"Hello, {params['name']}!"}
    }
}

resources = {
    'hello': {'id': 'hello', 'content': 'Hello, SLOP!'}
}

# In-memory storage
memory = {}

# CHAT
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('messages', [{}])[0].get('content', 'nothing')
    return jsonify({
        'message': {
            'role': 'assistant',
            'content': f'You said: {message}'
        }
    })

# TOOLS
@app.route('/tools', methods=['GET'])
def list_tools():
    return jsonify({'tools': list(tools.values())})

@app.route('/tools/<tool_id>', methods=['POST'])
def use_tool(tool_id):
    if tool_id not in tools:
        return jsonify({'error': 'Tool not found'}), 404
    return jsonify(tools[tool_id]['execute'](request.json))

# MEMORY
@app.route('/memory', methods=['POST'])
def store_memory():
    data = request.json
    memory[data['key']] = data['value']
    return jsonify({'status': 'stored'})

@app.route('/memory/<key>', methods=['GET'])
def get_memory(key):
    return jsonify({'value': memory.get(key)})

# RESOURCES
@app.route('/resources', methods=['GET'])
def list_resources():
    return jsonify({'resources': list(resources.values())})

@app.route('/resources/<resource_id>', methods=['GET'])
def get_resource(resource_id):
    if resource_id not in resources:
        return jsonify({'error': 'Resource not found'}), 404
    return jsonify(resources[resource_id])

# PAY
@app.route('/pay', methods=['POST'])
def pay():
    return jsonify({
        'transaction_id': f'tx_{int(datetime.now().timestamp())}',
        'status': 'success'
    })

def test_endpoints():
    """Test all SLOP endpoints"""
    base = 'http://localhost:5000'
    
    try:
        # Test chat
        print('📝 Testing chat...')
        chat = requests.post(f'{base}/chat', json={
            'messages': [{'content': 'Hello SLOP!'}]
        }).json()
        print(chat['message']['content'], '\n')

        # Test tools
        print('🔧 Testing tools...')
        calc = requests.post(f'{base}/tools/calculator', json={
            'expression': '2 + 2'
        }).json()
        print('2 + 2 =', calc['result'])

        greet = requests.post(f'{base}/tools/greet', json={
            'name': 'SLOP'
        }).json()
        print(greet['result'], '\n')

        # Test memory
        print('💾 Testing memory...')
        requests.post(f'{base}/memory', json={
            'key': 'test',
            'value': 'hello world'
        })
        memory = requests.get(f'{base}/memory/test').json()
        print('Stored value:', memory['value'], '\n')

        # Test resources
        print('📚 Testing resources...')
        hello = requests.get(f'{base}/resources/hello').json()
        print('Resource content:', hello['content'], '\n')

        # Test pay
        print('💰 Testing pay...')
        pay = requests.post(f'{base}/pay', json={
            'amount': 10
        }).json()
        print('Transaction:', pay['transaction_id'], '\n')

        print('✅ All tests passed!')
    except Exception as e:
        print('❌ Test failed:', str(e))

if __name__ == '__main__':
    import threading
    import time
    
    # Start server in a thread
    threading.Thread(target=app.run, daemon=True).start()
    
    # Wait for server to start
    print('✨ SLOP running on http://localhost:5000')
    time.sleep(1)
    print('🚀 Running tests...\n')
    
    # Run tests
    test_endpoints()