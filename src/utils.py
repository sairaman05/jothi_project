import os
import re
import json
import requests
from tqdm import tqdm

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None


class BaseLLM(object):
    def __init__(self, llm_name, ollama_url="http://localhost:11434"):
        self.llm_name = llm_name
        self.ollama_url = os.getenv("OLLAMA_HOST", ollama_url).rstrip("/")

        name_lower = llm_name.lower()
        if name_lower in ['llama3.1', 'llama3']:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for HuggingFace llama3.1 model.")
            self.llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            self.llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto')
        elif name_lower in ['gpt-4-turbo']:
            if AzureOpenAI is None:
                raise ImportError("openai package is required for gpt-4-turbo.")
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview"
            )
        else:
            self.ollama_model = llm_name
            try:
                r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if r.status_code == 200:
                    print(f"Connected to local Ollama host at {self.ollama_url} (model: {self.ollama_model})")
            except Exception as e:
                print(f"Warning: Could not connect to Ollama at {self.ollama_url}: {e}")

    
    def __generate_LLM__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.llm_model.device)

        terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id

        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=num_tokens_num,
            eos_token_id=terminators,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1]:]
        generated_text = self.llm_tokenizer.decode(response, skip_special_tokens=True)

        return generated_text

    def __generate_GPT__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_name, # Model deployment name      
                max_tokens = num_tokens_num,
                messages=messages
            )
        except Exception as e:
            return 'None'

        return response

    def __generate_Ollama__(self, query, num_tokens_num):
        url = f"{self.ollama_url}/api/generate"
        eff_tokens = max(num_tokens_num, 384) if "deepseek" in self.llm_name.lower() else num_tokens_num
        payload = {
            "model": getattr(self, "ollama_model", self.llm_name),
            "prompt": query,
            "stream": False,
            "options": {
                "num_predict": eff_tokens
            }
        }
        try:
            res = requests.post(url, json=payload, timeout=90)
            if res.status_code == 200:
                data = res.json()
                resp = data.get("response", "")
                if not resp and data.get("thinking"):
                    resp = data.get("thinking", "")
                return resp
            else:
                print(f"Ollama API Error {res.status_code}: {res.text}")
                return ""
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return ""

    
    def generate(self, query, new_tokens_num=512):
        name_lower = self.llm_name.lower()
        if name_lower in ['llama3.1', 'llama3']:
            return self.__generate_LLM__(query=query, num_tokens_num=new_tokens_num)
        elif name_lower in ['gpt-4-turbo']:
            return self.__generate_GPT__(query=query, num_tokens_num=new_tokens_num)
        else:
            return self.__generate_Ollama__(query=query, num_tokens_num=new_tokens_num)




class QADataset:
    def __init__(self, data, dir="dataset/"):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]
        question = data["question"]
        choices = [v for k, v in data["options"].items()]

        options = [" A: ", " B: ", " C: ", " D: "]

        text = question + "\n"
        for j in range(len(choices)):
            text += "{} {}\n".format(options[j], choices[j])

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        answer_content = choices[label_index]

        return {"text": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
    
class MedDDxLoader:
    def __init__(self, data, dir="dataset/"):
        benchmark = self.process_dataset(dir)
        self.data = data
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       

        benchmark = json.load(open(os.path.join(dir, "MedDDx.json")))
        
        data_dict = {'MedDDx':{}, 'MedDDx-Basic':{}, 'MedDDx-Intermediate':{}, 'MedDDx-Expert': {}}

        for idx, b in enumerate(benchmark):
            if b['sim_level_std'] > 0.04:
                data_dict['MedDDx-Basic'][idx] = b
            elif b['sim_level_std'] < 0.02:
                data_dict['MedDDx-Expert'][idx] = b
            else:
                data_dict['MedDDx-Intermediate'][idx] = b
            data_dict['MedDDx'][idx] = b
        
        return data_dict
        
    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
            
class AfrimedLoader:
    def __init__(self, data='mcq_expert', dir="dataset/"):
        print("data is {}".format(data))
        if data == 'AfrimedQA-MCQ':
            self.data = 'mcq_expert'
        elif data == 'AfrimedQA-SAQ':
            self.data = 'saq_expert' 
        
        benchmark = self.process_dataset(dir)
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        print("{} has {} queries".format(data, len(self.dataset)))
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       
        
        dataset_name = self.data
        datafile_name = "AfrimedQA_{}.json".format(dataset_name)

        print(datafile_name)

        if os.path.exists(os.path.join(dir, datafile_name)):
            dataset = json.load(open(os.path.join(dir, datafile_name)))
            return dataset
        else:
            from datasets import load_dataset
            options = [" A: ", " B: ", " C: ", " D: ", " E: ", " F: "]
            ds = load_dataset("intronhealth/afrimedqa_v2")['train']
            dataset = {dataset_name: {}}
            print("dataset is {}".format(dataset_name))
            
            for d in ds:
                print(d['question_type'])
                #if d['split'] == 'train':
                #    continue
                if d['tier'] != 'expert':
                    continue
                if d['question_type'] == 'mcq' and 'mcq' in dataset_name:
                    choices = [v for k, v in json.loads(d["answer_options"]).items()]
                
                    text = d['question_clean'].strip() + "\n"
                    for j in range(len(choices)):
                        text += "{} {}\n".format(options[j], choices[j])

                    label_index = int(d['correct_answer'][6])-1
                    answer = chr(ord('A') + label_index)
                    answer_content = choices[label_index]

                    idx = len(dataset['mcq_expert'])
                    dataset['mcq_expert'][idx] = {"query": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}
                if d['question_type'] == 'saq' and 'saq' in dataset_name:
                    
                    text = d['question_clean'].strip() + "\n"
                    answer = d['answer_rationale'].strip().replace('\n', ' ').replace('\r', '')

                    #label_index = int(d['correct_answer'][6])-1
                    #answer = chr(ord('A') + label_index)
                    #answer_content = choices[label_index]

                    idx = len(dataset['saq_expert'])
                    dataset['saq_expert'][idx] = {"query": text, "answer": answer, "answer_index": None, "answer_content": None}

            with open(os.path.join(dir, datafile_name), 'w') as f:
                json.dump(dataset, f, indent=2)

        return dataset

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        if self.data == 'saq_expert':
            label_index = answer
        else:        
            label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")


# =====================================================================
# Complete 11 Evaluation Metrics Suite (MedStreamMem Benchmark)
# =====================================================================

def calculate_accuracy(labels, predictions):
    """Metric 4: Accuracy (%) = (Correct Predictions / Total Questions) * 100%"""
    if not labels or len(labels) == 0:
        return 0.0
    correct = sum(1 for yt, yp in zip(labels, predictions) if str(yt).strip().upper() == str(yp).strip().upper())
    return round((correct / len(labels)) * 100.0, 2)


def calculate_precision_recall_f1(labels, predictions):
    """
    Metric 5: Precision (%) = (TP / (TP + FP)) * 100%
    Metric 6: Recall (%) = (TP / (TP + FN)) * 100%
    Metric 7: F1-Score (%) = 2 * (Precision * Recall) / (Precision + Recall)
    Computes Macro and Micro metrics across all classes.
    """
    if not labels or len(labels) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "class_breakdown": {}}

    y_true = [str(x).strip().upper() for x in labels]
    y_pred = [str(x).strip().upper() for x in predictions]

    valid_targets = sorted(list(set(c for c in y_true if c and c in ['A', 'B', 'C', 'D', 'E'])))
    classes = valid_targets if valid_targets else sorted(list(set(c for c in y_true if c)))
    if not classes:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "class_breakdown": {}}

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        p = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_precision = round(sum(precisions) / len(precisions), 2)
    macro_recall = round(sum(recalls) / len(recalls), 2)
    macro_f1 = (2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)) if (macro_precision + macro_recall) > 0 else 0.0
    macro_f1 = round(macro_f1, 2)

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "class_breakdown": {
            cls: {"precision": round(p, 2), "recall": round(r, 2), "f1": round(f, 2)}
            for cls, p, r, f in zip(classes, precisions, recalls, f1s)
        }
    }


def calculate_cache_hit_ratio(hits, misses):
    """Metric 8: Cache Hit Ratio (CHR %) = (N_hits / (N_hits + N_misses)) * 100%"""
    total = hits + misses
    if total <= 0:
        return 0.0
    return round((hits / total) * 100.0, 2)


def calculate_average_latency(chr_pct, l_hit=0.001, l_miss=12.4):
    """Metric 9: Average Latency (L_bar) = CHR * L_hit + (1 - CHR) * L_miss"""
    chr_ratio = chr_pct / 100.0
    l_bar = (chr_ratio * l_hit) + ((1.0 - chr_ratio) * l_miss)
    return round(l_bar, 4)


def calculate_high_trust_retention_ratio(entries, threshold=0.90):
    """Metric 10: High-Trust Retention Ratio (HTRR %) = (Sum I(tau_e >= 0.90) / K) * 100%"""
    if not entries:
        return 0.0
    count_high = sum(1 for e in entries if float(e.get("trust", 0.0)) >= threshold)
    return round((count_high / len(entries)) * 100.0, 2)


def calculate_memory_reduction_pct(ram_before_mb, ram_optimized_mb):
    """Metric 3: Memory Reduction % (MR %) = (1 - RAM_Optimized / RAM_Before) * 100%"""
    if ram_before_mb <= 0:
        return 0.0
    mr = (1.0 - (ram_optimized_mb / ram_before_mb)) * 100.0
    return round(max(0.0, min(100.0, mr)), 2)


def calculate_rouge_metrics(references, hypotheses):
    """Metric 11: ROUGE-1 / ROUGE-2 / ROUGE-L (Precision, Recall, F-Score)"""
    if not references or not hypotheses or len(references) != len(hypotheses):
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1_list, r2_list, rl_list = [], [], []
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            r1_list.append(scores['rouge1'].fmeasure * 100.0)
            r2_list.append(scores['rouge2'].fmeasure * 100.0)
            rl_list.append(scores['rougeL'].fmeasure * 100.0)
        return {
            "rouge1_f": round(sum(r1_list) / len(r1_list), 2),
            "rouge2_f": round(sum(r2_list) / len(r2_list), 2),
            "rougeL_f": round(sum(rl_list) / len(rl_list), 2)
        }
    except Exception as e:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0, "error": str(e)}


def compute_all_11_metrics(
    labels,
    predictions,
    ram_before_mb,
    ram_optimized_mb,
    cache_hits=0,
    cache_misses=0,
    avg_hit_latency=0.001,
    avg_miss_latency=12.4,
    memory_entries=None,
    saq_references=None,
    saq_hypotheses=None
):
    """
    Computes all 11 evaluation metrics defined in the MedStreamMem specification.
    """
    acc = calculate_accuracy(labels, predictions)
    prf = calculate_precision_recall_f1(labels, predictions)
    mr_pct = calculate_memory_reduction_pct(ram_before_mb, ram_optimized_mb)
    chr_pct = calculate_cache_hit_ratio(cache_hits, cache_misses)
    avg_lat = calculate_average_latency(chr_pct, avg_hit_latency, avg_miss_latency)
    htrr_pct = calculate_high_trust_retention_ratio(memory_entries or [], threshold=0.90)

    rouge_scores = {}
    if saq_references and saq_hypotheses:
        rouge_scores = calculate_rouge_metrics(saq_references, saq_hypotheses)

    return {
        "1_memory_before_mb": round(ram_before_mb, 2),
        "2_memory_optimized_mb": round(ram_optimized_mb, 2),
        "3_memory_reduction_pct": mr_pct,
        "4_accuracy_pct": acc,
        "5_precision_pct": prf["precision"],
        "6_recall_pct": prf["recall"],
        "7_f1_score_pct": prf["f1_score"],
        "8_cache_hit_ratio_pct": chr_pct,
        "9_average_latency_sec": avg_lat,
        "10_high_trust_retention_ratio_pct": htrr_pct,
        "11_rouge_scores": rouge_scores,
        "class_breakdown": prf.get("class_breakdown", {})
    }

