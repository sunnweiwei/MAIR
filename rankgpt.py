import tiktoken
from openai import OpenAI
import copy
import time
import os
import re


def run_gpt(messages, model="gpt-4o-mini"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    while True:
        try:
            completion = client.chat.completions.create(model=model, messages=messages)
            break
        except:
            time.sleep(0.1)
    return completion.choices[0].message.content


class RankGPT:
    def __init__(self, model='gpt-4o-mini', run_llm=None):
        """Updated version of RankGPT (https://arxiv.org/abs/2304.09542)
        Now support instruction input.
        model = RankGPT()
        results = model.rank('query', ['doc1', 'doc2', ...], prompt='xxx')
        """
        self.run_llm = run_llm or run_gpt
        self.model = model

    def get_prefix_prompt(self, query, num, instruction=None):
        if instruction is None:
            return [{'role': 'system',
                     'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                    {'role': 'user',
                     'content': f"Text Re-Ranking based on Ranking Instruction. I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                    {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
        else:
            return [{'role': 'system',
                     'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                    {'role': 'user',
                     'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}. Please use the following instructions to judge the relevance of the passages and prioritize the passages that best fits this instruction:\n{instruction}"},
                    {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    def get_post_prompt(self, query, num, instruction=None):
        if instruction is None:
            return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
        else:
            return f"Instruction: {instruction}\nSearch Query: {query}. \nRank the {num} passages above based on instruction. The passages should be listed in descending order using identifiers. The most relevant passages (i.e., best fits the requirement of instruction) should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def trunc(self, sentence, n):
        words = re.finditer(r'\S+|\s+', sentence)
        word_count = 0
        result = []
        for match in words:
            if match.group().strip():
                word_count += 1
            if word_count > n:
                break
            result.append(match.group())
        sentence = ''.join(result)
        if self.num_tokens_from_string(sentence) >= 512:
            return self.trunc(sentence, max(n - 32, n // 2))
        return sentence

    def create_permutation_instruction(self, query, candidates, instruction, rank_start=0, rank_end=100):
        num = len(candidates[rank_start: rank_end])
        messages = self.get_prefix_prompt(query, num, instruction)
        rank = 0
        for hit in candidates[rank_start: rank_end]:
            rank += 1
            content = hit['content']
            content = content.strip()
            content = self.trunc(content, 256)
            messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
            messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
        messages.append({'role': 'user', 'content': self.get_post_prompt(query, num, instruction)})
        return messages

    def clean_response(self, response: str):
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(self, response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(self, candidates, permutation, rank_start=0, rank_end=100):
        response = self.clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self.remove_duplicate(response)
        cut_range = copy.deepcopy(candidates[rank_start: rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            candidates[j + rank_start] = copy.deepcopy(cut_range[x])
        return candidates

    def permutation_pipeline(self, query, candidates, instruction=None, rank_start=0, rank_end=100):
        query = self.trunc(query, 256)
        messages = self.create_permutation_instruction(query, candidates, instruction, rank_start=rank_start,
                                                       rank_end=rank_end)
        permutation = self.run_llm(messages, model=self.model)
        results = self.receive_permutation(candidates, permutation, rank_start=rank_start, rank_end=rank_end)
        return results

    def sliding_window(self, query, candidates, instruction=None, rank_start=0, rank_end=None, window_size=20, step=10):
        candidates = copy.deepcopy(candidates)
        rank_end = rank_end or len(candidates)
        end_pos = rank_end
        start_pos = rank_end - window_size
        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            candidates = self.permutation_pipeline(query, candidates, instruction, start_pos, end_pos)
            end_pos = end_pos - step
            start_pos = start_pos - step
        return candidates

    def rank(self, query, documents, prompt=None):
        candidates = [{'corpus_id': i, 'content': doc} for i, doc in enumerate(documents)]
        results = self.sliding_window(query, candidates, instruction=prompt)
        results = [x | {'score': 1 / (i + 1)} for i, x in enumerate(results)]
        return results

    def parallel_rank(self, query, documents, prompt=None):
        # rank multiple queries in parallel
        raise NotImplemented

