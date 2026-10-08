import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Importing utils...", flush=True)
t0 = time.time()
from src.utils import BaseLLM, MedDDxLoader
print(f"Utils imported in {time.time()-t0:.2f}s", flush=True)

print("Importing actions...", flush=True)
t0 = time.time()
from action.generate import Generate
from action.review import Review
from action.answer import Answer
print(f"Actions imported in {time.time()-t0:.2f}s", flush=True)

class MockArgs:
    def __init__(self, **kwargs):
        self.llm_name = kwargs.get("llm_name", "llama3.2:3b")
        self.dataset = kwargs.get("dataset", "MedDDx-Basic")
        self.type = kwargs.get("type", "MCQ")
        self.max_round = kwargs.get("max_round", 1)
        self.is_revise = kwargs.get("is_revise", False)
        self.KG_name = kwargs.get("KG_name", "primeKG")
        self.weights_path = kwargs.get("weights_path", "fine_tuned_model/")
        self.use_cot = kwargs.get("use_cot", False)

print("Loading dataset...", flush=True)
loader = MedDDxLoader("MedDDx-Basic")
sample = loader[0]
print("Question:", sample["text"][:100], flush=True)
print("Ground truth answer:", sample.get("answer"), flush=True)

print("\nInitializing BaseLLM(llama3.2:3b)...", flush=True)
t0 = time.time()
args = MockArgs(llm_name="llama3.2:3b")
llm = BaseLLM("llama3.2:3b")
print(f"BaseLLM init in {time.time()-t0:.2f}s", flush=True)

print("\nTesting LLM direct generation: 'Say hello in 5 words'...", flush=True)
t0 = time.time()
test_resp = llm.generate("Say hello in 5 words", new_tokens_num=20)
print(f"Direct LLM response ({time.time()-t0:.2f}s): {repr(test_resp)}", flush=True)

print("\nInitializing Generate...", flush=True)
t0 = time.time()
gen = Generate(llm, args)
print(f"Generate init in {time.time()-t0:.2f}s", flush=True)

print("Initializing Review...", flush=True)
t0 = time.time()
rev = Review(llm, args)
print(f"Review init in {time.time()-t0:.2f}s", flush=True)

print("Initializing Answer...", flush=True)
t0 = time.time()
ans = Answer(llm, use_cot=False)
print(f"Answer init in {time.time()-t0:.2f}s", flush=True)

print("\nStep 1: gen.call()...", flush=True)
t0 = time.time()
gen_triplets = gen.call(sample["text"])
print(f"gen.call() finished in {time.time()-t0:.2f}s with triplets: {gen_triplets}", flush=True)

print("\nStep 2: rev.call()...", flush=True)
t0 = time.time()
fil_triplets, scores = rev.call(gen_triplets, sample["text"])
print(f"rev.call() finished in {time.time()-t0:.2f}s with filtered: {fil_triplets}", flush=True)

print("\nStep 3: ans.call()...", flush=True)
t0 = time.time()
raw_ans = ans.call(fil_triplets, sample["text"])
print(f"ans.call() finished in {time.time()-t0:.2f}s: {raw_ans}", flush=True)
