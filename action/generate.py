from transformers import set_seed
import re
import json
import logging
from src.promptTemplate import triplet_prompt_template, triplet_prompt_template_for_binary_or_maybe, generation_prompt_template


class QueryAnalysis(object):
    def __init__(self, args, model_name, device):
        self.args = args
        self.model_name = model_name
        self.device = device

    def query_identify(self, query):
        
        items = query.strip().split('\n')
        question = items[0]
        options = items[1:]

        content = []
        for o in options:
            content.append(o.strip()[2:].strip())

        if len(content) == 2 and 'yes' in content and 'no' in content:
            query_type = 1 
        elif len(content) == 3 and 'maybe' in content and 'yes' in content and 'no' in content:
            query_type = 2
        else:
            query_type = 3
        
        return query_type, question, content
    
class TripletExtraction(object):
    def __init__(self, llm, args, model_name, device):
        self.llm = llm
        self.args = args
        self.model_name = model_name
        self.device = device
        set_seed(42)
        self.q_type = QueryAnalysis(args, model_name, device)
    
    def check_entities(self, keys_text):
        try:
            text = re.sub(r'```(?:json)?', '', keys_text).replace('```', '').strip()
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                keys_dict = json.loads(json_str)
            else:
                pattern = r"\{(.*?)\}"
                matches = re.findall(pattern, text.replace("\n", ""))
                if not matches:
                    return ""
                keys_dict = json.loads("{" + matches[0] + "}")

            mt_list = None
            for k in keys_dict:
                if 'medical_terminolog' in k.lower() or 'terminolog' in k.lower() or 'entities' in k.lower():
                    mt_list = keys_dict[k]
                    break

            if not mt_list:
                return ""

            if isinstance(mt_list, list):
                mt = list(set(mt_list))
                return ', '.join(str(item) for item in mt)
            elif isinstance(mt_list, str):
                return mt_list
            return ""
        except Exception:
            return ""

    def check_triplets(self, keys_text):
        try:
            text = re.sub(r'```(?:json)?', '', keys_text).replace('```', '').strip()
            text = text.replace('(', '[').replace(')', ']')
            
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                keys_dict = json.loads(json_str)
            else:
                pattern = r"\{(.*?)\}"
                matches = re.findall(pattern, text.replace("\n", ""))
                if not matches:
                    return ""
                new_match = matches[0]
                if new_match and new_match[0] == '[':
                    new_match = '"Triplets": ' + new_match
                keys_dict = json.loads("{" + new_match + "}")

            triplets_list = None
            for k in keys_dict:
                if 'triplet' in k.lower():
                    triplets_list = keys_dict[k]
                    break

            if not triplets_list or not isinstance(triplets_list, list):
                return ""

            return list(triplets_list)
        except Exception:
            return ""

    def generated_related_entities(self, query):

        q_type, question, answer_option = self.q_type.query_identify(query)
        mt = ""
        count = 0
        while mt == "" and count < 5:
            generated_text = self.llm.generate(generation_prompt_template.replace('{query}', question), 128)
            mt = self.check_entities(generated_text)
            count += 1

        logging.info(mt)
        
        all_triplets = []
        
        
        if q_type == 3:
            for a in answer_option:
                generated_triplets = self.llm.generate(triplet_prompt_template.replace('{query_stem}', question).replace('{mt}', mt).replace('{option}', a), 256)
                triplets = self.check_triplets(generated_triplets)
                           
                if triplets == "" or len(triplets) == 0:
                    continue
                else:
                    all_triplets.extend(triplets)
        else:
            generated_triplets = self.llm.generate(triplet_prompt_template_for_binary_or_maybe.replace('{query_stem}', query).replace('{mt}', mt), 256)
            all_triplets = self.check_triplets(generated_triplets)
        

        if all_triplets == "" or len(all_triplets) == 0:
            return "There is no triplets related to input query"

        logging.info(all_triplets)        
        return str(all_triplets)

class Generate(object):
    def __init__(self, llm, args) -> None:
        super().__init__()
        self.class_name = 'Extract_Triplets'
        self.class_desc = 'Using this action to extract triplets related to query.'
        self.llm = llm
        self.tripleExtraction = TripletExtraction(llm = self.llm, args = args, model_name=args.llm_name, device='auto')
        self.args = args
        
    def call(self, query):
        triplets = self.tripleExtraction.generated_related_entities(query)
        return triplets
