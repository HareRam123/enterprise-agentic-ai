Small LangGraph Math Agent (Python)

Setup
1. Create venv (already created):
   /usr/local/bin/python3 -m venv .venv
2. Activate venv:
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Add your secrets in .env:
   OPENAI_API_KEY=your_real_key
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=openai/gpt-4o-mini

OpenRouter note
- If you use OpenRouter, keep your key in OPENAI_API_KEY and set OPENAI_BASE_URL.
- OPENAI_MODEL should be a provider/model id (example: openai/gpt-4o-mini).

What this agent does
- Uses a LangGraph reasoning/action loop.
- Calls math tools for addition, subtraction, multiplication, and division.
- Returns a direct answer after tool execution completes.

Notes
- .env is ignored by git.
- .env.example is safe to commit and share.

Run
python agent.py

Visualize the compiled graph
python agent.py --graph
python agent.py --graph-file graph.mmd
python agent.py --graph-png graph.png

Example
You: what is (12 + 8) * 3?
Agent: The answer is 60.
