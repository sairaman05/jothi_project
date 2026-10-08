from typing import List, Optional, Dict
from transformers import set_seed
import re
import json
import logging
from src.promptTemplate import answer_generation_prompt_template


def extract_mcq_options(query: str) -> Dict[str, str]:
    """
    Extracts MCQ options (e.g., A: ..., B: ...) from the query string.
    """
    options = {}
    matches = re.findall(r'([A-E])[:\.]\s*([^\n]+)', query)
    for letter, text in matches:
        options[letter.upper()] = text.strip()
    return options


def format_mcq_response(outputs: str, options: Dict[str, str]) -> str:
    """
    Extracts the predicted MCQ letter from LLM output (including CoT and DeepSeek-R1 outputs).
    """
    if not options:
        return outputs

    cleaned_output = outputs
    # If output contains think block (e.g. DeepSeek-R1), focus search on post-think conclusion first
    if "</think>" in outputs:
        post_think = outputs.split("</think>")[-1].strip()
        if post_think:
            cleaned_output = post_think

    selected_letter = None

    # Search patterns from most specific to general
    patterns = [
        r'(?:Final\s+Answer|Correct\s+Answer|The\s+correct\s+choice\s+is|Answer)[\s:]*(?:Option\s*)?([A-E])\b',
        r'Option\s*([A-E])\s*(?:is\s+correct|is\s+the\s+answer)',
        r'(?:Answer["\s:]+)?\b([A-E])\b',
        r'\b([A-E])[\.:\)]'
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, cleaned_output, re.IGNORECASE))
        if matches:
            # Pick the last occurrence (often the concluding statement in CoT)
            selected_letter = matches[-1].group(1).upper()
            break

    # Fallback to whole output search if not found in post-think
    if not selected_letter and "</think>" in outputs:
        for pat in patterns:
            matches = list(re.finditer(pat, outputs, re.IGNORECASE))
            if matches:
                selected_letter = matches[-1].group(1).upper()
                break

    if selected_letter and selected_letter in options:
        option_text = options[selected_letter]
        return f"Answer: Option {selected_letter} - {option_text}"

    return outputs


class Answer(object):
    """
    Answer Generator with optional Chain-of-Thought (CoT) Clinical Reasoning.
    Supports local Ollama models (llama3.2:3b, llama3:latest, deepseek-r1:14b).
    """
    def __init__(self, llm, use_cot: bool = False) -> None:
        super().__init__()
        self.class_name = 'Answer_Generator'
        self.class_desc = 'Using this action to generate answer directly with optional Chain-of-Thought.'
        self.llm = llm
        self.use_cot = use_cot
        set_seed(42)

    def call(self, filtered_triplets, query, use_cot: Optional[bool] = None):
        cot_active = self.use_cot if use_cot is None else use_cot
        options = extract_mcq_options(query)
        is_mcq = len(options) > 0 or bool(re.search(r'\b[A-E]:', query))

        if is_mcq:
            if cot_active:
                # Phase 2 Chain-of-Thought (CoT) Clinical Reasoning Prompt
                cot_prompt = (
                    "You are an expert clinical diagnostician. Use step-by-step Chain-of-Thought (CoT) reasoning to evaluate the medical scenario and select the correct clinical option.\n\n"
                    f"Verified Knowledge Triplets:\n{filtered_triplets}\n\n"
                    f"Clinical Question and Options:\n{query}\n\n"
                    "Instructions for Chain-of-Thought Reasoning:\n"
                    "1. Clinical Presentation: Identify symptoms, vitals, and pathology.\n"
                    "2. Triplet Correlation: Map knowledge triplets to differential diagnoses.\n"
                    "3. Option Evaluation: Systematically evaluate options and eliminate incorrect choices.\n"
                    "4. Final Conclusion: State the single correct option in the exact format: 'Answer: Option [LETTER]'\n\n"
                    "Clinical Reasoning and Final Answer:"
                )
                raw_output = self.llm.generate(cot_prompt, new_tokens_num=256)
            else:
                prompt = answer_generation_prompt_template.replace('{t}', str(filtered_triplets)).replace('{q}', query)
                raw_output = self.llm.generate(prompt, new_tokens_num=80)

            formatted_answer = format_mcq_response(raw_output, options)
            return formatted_answer
        else:
            if cot_active:
                open_prompt = (
                    "You are an expert clinical physician. Use step-by-step clinical reasoning to answer the following medical question comprehensively.\n\n"
                    f"Verified Knowledge Triplets:\n{filtered_triplets}\n\n"
                    f"Clinical Query:\n{query}\n\n"
                    "Provide a clear, evidence-based clinical answer:\nAnswer:"
                )
            else:
                open_prompt = (
                    f"You are an expert medical assistant. Based on the provided medical knowledge triplets and your clinical knowledge, "
                    f"provide a direct, accurate, and comprehensive answer to the user's question. Do NOT output option letters like 'Answer: C' unless options were provided in the question.\n\n"
                    f"Verified Knowledge Triplets: {filtered_triplets}\n\n"
                    f"User Question: {query}\n\n"
                    f"Answer:"
                )
            outputs = self.llm.generate(open_prompt, new_tokens_num=256)
            return outputs


class AnswerOpen(object):
    def __init__(self, llm) -> None:
        super().__init__()
        self.class_name = 'Answer_Open_Generator'
        self.class_desc = 'Using this action to generate open-ended answers directly.'
        self.llm = llm
        set_seed(42)

    def call(self, filtered_triplets, query):
        open_prompt = (
            f"Based on the provided medical knowledge triplets and your clinical knowledge, "
            f"provide a direct, accurate, and comprehensive answer to the user's question.\n\n"
            f"Verified Knowledge Triplets: {filtered_triplets}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )
        outputs = self.llm.generate(open_prompt, new_tokens_num=256)
        return outputs