import logging
from src.utils import BaseLLM
from .generate import TripletExtraction
from .revise import Revise


class Review(object):
    def __init__(self, llm, args) -> None:
        super().__init__()
        self.class_name = 'Review'
        self.llm = llm
        self.args = args
        self.action_desc = 'Using this action to score generated triplets.'
        self.use_llm_eval = False

        if hasattr(self.llm, 'llm_model') and self.llm.llm_model is not None:
            try:
                from .inference_review import ReviewInfer
                self.model = ReviewInfer(model = self.llm.llm_model, tokenizer=self.llm.llm_tokenizer, model_weights = args.weights_path)
            except Exception as e:
                logging.info(f"Using LLM-prompted verification for Review step: {e}")
                self.use_llm_eval = True
        else:
            self.use_llm_eval = True

        self.is_revise = args.is_revise
        if self.is_revise == True: 
            self.revise = Revise(self.llm)
        self.max_round = args.max_round
        self.triple_generator = TripletExtraction(llm=self.llm, args=args, model_name=args.llm_name, device='cuda')
    
    def check_triplets(self, keys_text):
        if isinstance(keys_text, list):
            return keys_text
        match = str(keys_text).replace('[', '').replace(']', '').replace('\'', '')
        triplets_list = match.split(',')
        split_length = 3
        split_lists = [triplets_list[i:i+split_length] for i in range(0, int((len(triplets_list)/split_length))*split_length , split_length)]
        return split_lists

    def score_triplet(self, t, query=""):
        if not self.use_llm_eval:
            try:
                return self.model.score(t)
            except Exception as e:
                logging.info(f"ReviewInfer error on {t}: {e}. Falling back to LLM scoring.")

        head = t[0] if len(t) > 0 else ""
        rel = t[1] if len(t) > 1 else ""
        tail = t[2] if len(t) > 2 else ""
        prompt = f"Given medical context: {query}\nDetermine if the triplet ({head}, {rel}, {tail}) is clinically accurate and valid. Reply ONLY with 'True' or 'False'."
        res = self.llm.generate(prompt, 10).strip()
        if 'true' in res.lower():
            return 'True', 0.95
        else:
            return 'False', 0.10
    
    def output_format(self, output):
        return str(output)
    
    def call(self, triplets, query):
        triplet_list = self.check_triplets(triplets)
        if isinstance(triplet_list, list) and len(triplet_list) > 5:
            triplet_list = triplet_list[:5]
        scores = []
        select_triplets = []
        
        for t in triplet_list:
            t = [str(i).strip() for i in t]
            if len(t) < 3:
                continue
            classification, prob = self.score_triplet(t, query)
            logging.info("{} is {}".format(t, classification))
            if classification == 'True':
                select_triplets.append(t)
                scores.append(prob)
            elif classification == 'False' and self.is_revise:
                temp = [t]
                round_num = 0
                while round_num < self.max_round:
                    logging.info("current round is {} and triplets are {}".format(round_num, temp))
                    
                    modified_triple = self.revise.call(temp, query)
                    modified_triple_list = self.check_triplets(modified_triple)
                    
                    for m in modified_triple_list:
                        m = [str(e).strip() for e in m]
                        if len(m) < 3:
                            continue
                        m_class, m_prob = self.score_triplet(m, query)
                        logging.info("revised triplet {} is {}".format(m, m_class, m_prob))
                        if m_class == 'True':
                            select_triplets.append(m)
                            scores.append(m_prob)
                            round_num = self.max_round
                            break
                        temp.append(m)
                    round_num += 1

        return self.output_format(select_triplets), scores