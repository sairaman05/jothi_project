from typing import List
from transformers import set_seed
import re
import json
import logging
from src.promptTemplate import answer_generation_prompt_template


def extract_mcq_options(query):
    options = {}
    matches = re.findall(r'([A-E])[:\.]\s*([^\n]+)', query)
    for letter, text in matches:
        options[letter.upper()] = text.strip()
    return options


def format_mcq_response(outputs, options):
    if not options:
        return outputs

    selected_letter = None
    # Try finding "Answer": "B" or "Answer: B" or option letter B
    match = re.search(r'(?:Answer["\s:]+)?\b([A-E])\b', outputs, re.IGNORECASE)
    if match:
        selected_letter = match.group(1).upper()

    if selected_letter and selected_letter in options:
        option_text = options[selected_letter]
        return f"Answer: Option {selected_letter} - {option_text}"

    return outputs


class Answer(object):
    def __init__(self, llm) -> None:
        super().__init__()
        self.class_name = 'Answer_Generator'
        self.class_desc = 'Using this action to generate answer directly.'
        self.llm = llm
        set_seed(42)
        
    def call(self, filtered_triplets, query):
        options = extract_mcq_options(query)
        is_mcq = len(options) > 0 or bool(re.search(r'\b[A-E]:', query))
        
        if is_mcq:
            prompt = answer_generation_prompt_template.replace('{t}', str(filtered_triplets)).replace('{q}', query)
            raw_output = self.llm.generate(prompt, new_tokens_num=60)
            formatted_answer = format_mcq_response(raw_output, options)
            return formatted_answer
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