import argparse
import os
import time
import json
import logging
import re
from tqdm import tqdm
from transformers import set_seed
from src.utils import QADataset, MedDDxLoader, BaseLLM, AfrimedLoader
from src.memory import MedStreamMem
from action.generate import Generate
from action.review import Review
from action.answer import Answer

set_seed(42)


class KGARevion(object):
    def __init__(self, args):
        super().__init__()
        self.agent_name = "KGARevion"
        self.role = """You can answer questions by choosing Extract_Triplets, KnowledgeGraph_Classifier and Answer_Generator actions. Finish it if you find answer."""
        self.args = args
        self.llm = BaseLLM(args.llm_name)
        self.triplets_generator = Generate(self.llm, args)
        
        if args.llm_name == 'gpt-4-turbo':
            self.review_llm = BaseLLM('llama3.1')
        else:
            self.review_llm = self.llm
            
        self.classifier = Review(self.review_llm, args)
        self.answer_generator = Answer(self.llm)

        # Initialize Bounded Memory (MedStreamMem)
        self.enable_memory = getattr(args, 'enable_memory', True)
        if self.enable_memory:
            capacity = getattr(args, 'memory_capacity', 50)
            default_trust = getattr(args, 'default_trust', 0.95)
            self.memory = MedStreamMem(capacity=capacity, default_trust=default_trust)
            print(f"[MedStreamMem] Bounded memory initialized with Capacity={capacity}, Trust={default_trust}")
        else:
            self.memory = None

    def call(self, query):
        logging.info(f"Query: {query}")
        
        # 1. Check MedStreamMem Memory Cache
        if self.memory is not None:
            cached_result = self.memory.get(query)
            if cached_result is not None:
                print(f"\n[MedStreamMem CACHE HIT] Query found in memory! Priority Scores updated.")
                logging.info(f"[MedStreamMem CACHE HIT] Returning cached answer.")
                return cached_result.get("answer", ""), cached_result

        # 2. Triplet Extraction & Verification Pipeline
        generated_triplets = self.triplets_generator.call(query)
        filtered_triplets, scores = self.classifier.call(generated_triplets, query)
        answer = self.answer_generator.call(filtered_triplets, query)
        
        logging.info("Filtered triplets: {}".format(filtered_triplets))
        logging.info("Answer: {}".format(answer))

        result_data = {
            "answer": answer,
            "generated_triplets": generated_triplets,
            "filtered_triplets": filtered_triplets,
            "review_scores": scores
        }

        # 3. Store in MedStreamMem Bounded Memory
        if self.memory is not None:
            inserted, evicted_key = self.memory.put(query, result_data, trust=self.args.default_trust)
            if evicted_key:
                print(f"[MedStreamMem EVICTION] Memory full. Evicted entry with lowest Priority Score: '{evicted_key[:40]}...'")
                logging.info(f"[MedStreamMem EVICTION] Evicted: {evicted_key}")

        return answer, result_data


def main(args):
    set_seed(42)

    import gc
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f"{args.dataset}_test_case_study.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    bioKG_agent = KGARevion(args=args)

    # Restore memory snapshot if available
    snapshot_filepath = os.path.join(args.output_dir, "memory_snapshot.json")
    if bioKG_agent.memory and os.path.exists(snapshot_filepath):
        if bioKG_agent.memory.load_snapshot(snapshot_filepath):
            print(f"[MedStreamMem] Restored memory snapshot from: {snapshot_filepath}")

    # Load dataset
    if args.dataset in ['MedDDx', 'MedDDx-Basic', 'MedDDx-Intermediate', 'MedDDx-Expert']:
        data = MedDDxLoader(args.dataset)
    elif args.dataset in ['AfrimedQA-MCQ']:
        data = AfrimedLoader(args.dataset)
    else:
        data = QADataset(args.dataset)

    accurate_sample_idx = []
    response_all = []
    structured_query_results = []
    processed_indices = set()

    # Resume support: check for partial run JSON
    partial_filepath = os.path.join(args.output_dir, f"partial_{args.dataset}_{args.llm_name.replace(':', '_')}.json")
    if getattr(args, 'resume', True):
        # Look for partial file or run JSON files
        candidate_files = [partial_filepath]
        for fn in os.listdir(args.output_dir):
            if fn.startswith(f"run_{args.dataset}_") and fn.endswith(".json"):
                candidate_files.append(os.path.join(args.output_dir, fn))

        for cfile in candidate_files:
            if os.path.exists(cfile):
                try:
                    with open(cfile, "r", encoding="utf-8") as f:
                        prev_data = json.load(f)
                    prev_queries = prev_data.get("queries", [])
                    for qrec in prev_queries:
                        s_idx = qrec.get("sample_index")
                        if s_idx is not None and s_idx not in processed_indices:
                            processed_indices.add(s_idx)
                            structured_query_results.append(qrec)
                            response_all.append(qrec.get("raw_response", ""))
                            if qrec.get("is_correct"):
                                accurate_sample_idx.append(s_idx)
                    print(f"[Resume Engine] Loaded {len(processed_indices)} previously processed samples from {cfile}")
                except Exception as e:
                    print(f"[Resume Engine] Could not load partial data from {cfile}: {e}")

    for idx, d in tqdm(enumerate(data), total=len(data), desc=f'Evaluating dataset ({args.dataset})'):
        if idx in processed_indices:
            continue  # Skip already processed sample

        query = d.get('text', '')
        label = d.get('answer', '')

        start_t = time.time()
        response, result_meta = bioKG_agent.call(query)
        elapsed_t = time.time() - start_t
        
        response_clean = response.strip().replace('\n', '').replace('\"', '')
        logging.info(f'Index: {idx}, Response: {response_clean}, Label: {label}')

        predict_answer = 'None'
        answer_index = response_clean.find("Answer: ")
        if answer_index != -1:
            predict_answer_raw = response_clean[answer_index + len("Answer: "):].strip()
            match_letter = re.search(r'\b([A-E])\b', predict_answer_raw)
            if match_letter:
                predict_answer = match_letter.group(1).upper()
            elif predict_answer_raw:
                predict_answer = predict_answer_raw[0].upper()

        if predict_answer not in ['A', 'B', 'C', 'D', 'E'] and label in response_clean:
            predict_answer = label
        predict_answer = predict_answer.strip()

        is_correct = (predict_answer == label)
        if is_correct:
            accurate_sample_idx.append(idx)

        response_all.append(response_clean)
        processed_indices.add(idx)

        # Telemetry record for structured JSON output
        query_record = {
            "sample_index": idx,
            "query": query,
            "ground_truth": label,
            "predicted_answer": predict_answer,
            "is_correct": is_correct,
            "raw_response": response,
            "elapsed_seconds": round(elapsed_t, 2),
            "generated_triplets": result_meta.get("generated_triplets", ""),
            "filtered_triplets": result_meta.get("filtered_triplets", ""),
            "review_scores": result_meta.get("review_scores", [])
        }
        structured_query_results.append(query_record)

        # Save partial progress and memory snapshot every 3 samples
        if len(processed_indices) % 3 == 0:
            partial_data = {
                "dataset": args.dataset,
                "llm_name": args.llm_name,
                "total_processed": len(processed_indices),
                "correct_samples": len(accurate_sample_idx),
                "queries": structured_query_results
            }
            with open(partial_filepath, "w", encoding="utf-8") as f:
                json.dump(partial_data, f, indent=2)
            if bioKG_agent.memory:
                bioKG_agent.memory.save_snapshot(snapshot_filepath)

    # Calculate metrics
    metrics_value = 0.0
    if args.type == 'SAQ':
        from rouge_score import rouge_scorer
        correct_predictions = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for idx, r in enumerate(response_all):
            results = scorer.score(r, data[idx]['answer'])
            correct_predictions.append([results['rouge1'].precision, results['rouge1'].recall, results['rouge1'].fmeasure])
        import numpy as np
        correct_predictions = np.array(correct_predictions)
        correct_predictions = np.sum(correct_predictions, axis=0)
        metrics_value = correct_predictions / max(1, len(response_all))
        print(f"Rouge: {metrics_value[0]:.2%} {metrics_value[1]:.2%} {metrics_value[2]:.2%}")
    elif args.type == 'MCQ':
        metrics_value = len(accurate_sample_idx) / max(1, len(response_all))
        print(f"Final Accuracy: {metrics_value:.2%} ({len(accurate_sample_idx)}/{len(response_all)})")

    timestamp = int(time.time())
    
    # Save final structured JSON output file
    json_filename = f"run_{args.dataset}_{args.llm_name.replace(':', '_')}_{timestamp}.json"
    json_filepath = os.path.join(args.output_dir, json_filename)
    
    run_output_data = {
        "run_info": {
            "timestamp": timestamp,
            "dataset": args.dataset,
            "llm_name": args.llm_name,
            "max_round": args.max_round,
            "is_revise": args.is_revise,
            "enable_memory": args.enable_memory,
            "memory_capacity": args.memory_capacity,
            "default_trust": args.default_trust,
            "accuracy": metrics_value if isinstance(metrics_value, float) else metrics_value.tolist()
        },
        "metrics": {
            "total_samples": len(response_all),
            "correct_samples": len(accurate_sample_idx),
            "accuracy": metrics_value if isinstance(metrics_value, float) else metrics_value.tolist()
        },
        "queries": structured_query_results
    }

    if bioKG_agent.memory is not None:
        run_output_data["memory_telemetry"] = bioKG_agent.memory.get_telemetry()
        bioKG_agent.memory.save_snapshot(snapshot_filepath)
        print(f"[MedStreamMem] Memory snapshot saved to: {snapshot_filepath}")

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(run_output_data, f, indent=2)
    print(f"[Output Saved] Full JSON run output saved to: {json_filepath}")

    # Remove partial file on clean completion
    if os.path.exists(partial_filepath):
        try:
            os.remove(partial_filepath)
        except Exception:
            pass

    # Legacy text report file
    txt_filepath = os.path.join(args.output_dir, f"{args.dataset}_{args.llm_name.replace(':', '_')}_summary.txt")
    with open(txt_filepath, 'w', encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)
        f.write('\n')
        f.write(f"accuracy: {metrics_value}\n")
        f.write("correct task ids: " + "\t".join(str(a) for a in accurate_sample_idx) + "\n\n")
        for idx, r in enumerate(response_all):
            f.write(f"{idx}: {r}\n")
    print(f"[Output Saved] Summary report saved to: {txt_filepath}")


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description="KGARevion with MedStreamMem Bounded Memory & Ollama Support")
    parser.add_argument("--dataset", default='MedDDx-Basic', choices=['mmlu', 'medqa', 'pubmedqa', 'bioasq', 'MedDDx', 'MedDDx-Basic', 'MedDDx-Intermediate', 'MedDDx-Expert', 'afrimedqa_v2', 'AfrimedQA-MCQ', 'AfrimedQA-SAQ'], type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--type", type=str, default='MCQ', choices=['MCQ', 'SAQ'])
    parser.add_argument("--max_round", type=int, default=1)
    parser.add_argument("--is_revise", type=bool, default=True)
    parser.add_argument("--KG_name", default='primeKG', choices=['UMLS', 'primeKG', 'ogb-biokg'], type=str)
    parser.add_argument("--llm_name", default='llama3.2:3b', type=str, help="LLM model name (e.g., llama3.2:3b, llama3.1, gpt-4-turbo)")
    parser.add_argument("--weights_path", type=str, default='fine_tuned_model/')
    parser.add_argument("--enable_memory", type=bool, default=True, help="Enable MedStreamMem bounded memory buffer")
    parser.add_argument("--memory_capacity", type=int, default=50, help="Maximum items allowed in MedStreamMem buffer")
    parser.add_argument("--default_trust", type=float, default=0.95, help="Default source trust score (tau_trust)")
    parser.add_argument("--output_dir", type=str, default='results', help="Directory to save output JSON and logs")
    parser.add_argument("--resume", type=bool, default=True, help="Resume execution from saved partial results and memory snapshots")
    
    args = parser.parse_args()
    main(args)
