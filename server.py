import argparse
import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from src.utils import BaseLLM
from src.memory import MedStreamMem
from KGARevion import KGARevion

app = Flask(__name__, static_folder='ui')

# Global agent & args instance
agent = None
agent_args = None


def init_agent(args):
    global agent, agent_args
    agent_args = args
    print(f"Initializing KGARevion with LLM={args.llm_name}, Memory Capacity={args.memory_capacity}, Default Trust={args.default_trust}")
    agent = KGARevion(args)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.json or {}
    query = data.get('query', '').strip()
    trust = data.get('trust', agent_args.default_trust if agent_args else 0.95)

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        # Check if query hits memory cache first
        cache_hit = False
        if agent.memory is not None:
            cached = agent.memory.get(query)
            if cached is not None:
                cache_hit = True

        start_evictions = len(agent.memory.eviction_history) if agent.memory else 0

        # Run KGARevion pipeline
        answer, meta = agent.call(query)

        end_evictions = len(agent.memory.eviction_history) if agent.memory else 0
        new_eviction = (end_evictions > start_evictions)

        memory_telemetry = agent.memory.get_telemetry() if agent.memory else {}

        # Save updated memory snapshot
        if agent.memory:
            os.makedirs("results", exist_ok=True)
            agent.memory.save_snapshot("results/memory_snapshot.json")

        filtered_triplets = meta.get("filtered_triplets", [])
        if isinstance(filtered_triplets, str):
            try:
                import ast
                filtered_triplets = ast.literal_eval(filtered_triplets)
            except Exception:
                pass

        generated_triplets = meta.get("generated_triplets", [])
        if isinstance(generated_triplets, str):
            try:
                import ast
                generated_triplets = ast.literal_eval(generated_triplets)
            except Exception:
                pass

        response_payload = {
            "query": query,
            "answer": answer,
            "cache_hit": cache_hit,
            "new_eviction": new_eviction,
            "generated_triplets": generated_triplets,
            "filtered_triplets": filtered_triplets,
            "review_scores": meta.get("review_scores", []),
            "medical_terminologies": meta.get("medical_terminologies", ""),
            "memory": memory_telemetry
        }
        return jsonify(response_payload), 200

    except Exception as e:
        logging.exception("Error in /api/chat endpoint")
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory', methods=['GET'])
def get_memory_endpoint():
    if agent and agent.memory:
        return jsonify(agent.memory.get_telemetry()), 200
    return jsonify({'capacity': 0, 'current_size': 0, 'entries': [], 'eviction_history': []}), 200


@app.route('/api/memory/clear', methods=['POST', 'OPTIONS'])
def clear_memory_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if agent and agent.memory:
        agent.memory.buffer.clear()
        agent.memory.current_step = 0
        agent.memory.eviction_history.clear()
        agent.memory.save_snapshot("results/memory_snapshot.json")
        return jsonify({'status': 'cleared', 'telemetry': agent.memory.get_telemetry()}), 200
    return jsonify({'status': 'no_memory'}), 200


@app.route('/api/runs', methods=['GET'])
def list_runs_endpoint():
    results_dir = "results"
    runs = []
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.startswith("run_") and fname.endswith(".json"):
                fpath = os.path.join(results_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        runs.append(json.load(f))
                except Exception:
                    pass
    return jsonify(runs), 200


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_ui(path):
    ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui')
    if path and os.path.exists(os.path.join(ui_dir, path)):
        return send_from_directory(ui_dir, path)
    return send_from_directory(ui_dir, 'react_dashboard.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask Backend Server for KGARevion & MedStreamMem React UI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--llm_name", type=str, default="llama3.2:3b")
    parser.add_argument("--dataset", type=str, default="MedDDx-Basic")
    parser.add_argument("--type", type=str, default="MCQ")
    parser.add_argument("--max_round", type=int, default=1)
    parser.add_argument("--is_revise", type=bool, default=True)
    parser.add_argument("--weights_path", type=str, default="fine_tuned_model/")
    parser.add_argument("--enable_memory", type=bool, default=True)
    parser.add_argument("--memory_capacity", type=int, default=5)
    parser.add_argument("--default_trust", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()
    init_agent(args)

    print(f"\n🚀 Server running at http://localhost:{args.port}/")
    print(f"💬 React Interactive Chat ready at http://localhost:{args.port}/")
    print(f"🧠 MedStreamMem Memory Bounding Formula Active with Capacity={args.memory_capacity}\n")
    app.run(host=args.host, port=args.port, debug=False)
