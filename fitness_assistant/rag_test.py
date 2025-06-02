#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import minsearch
#import os 
df = pd.read_csv('../data/data.csv')
documents = df.to_dict(orient='records')
index = minsearch.Index(text_fields=['exercise_name', 'type_of_activity', 'type_of_equipment',
       'body_part', 'type', 'muscle_groups_activated', 'instructions'],
                        keyword_fields=[])
index.fit(documents)


# ## RAG flow ##

import os
from openai import OpenAI
print(os.environ['OPENAI_API_KEY'])
client = OpenAI()


def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


prompt_template = """
You're a fitness insrtuctor. Answer the QUESTION based on the CONTEXT from our exercises database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instructions: {instructions}
""".strip()

def build_prompt(query, search_results):
    context = ""
    
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[54]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


# In[55]:


def rag(query, model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer


# In[56]:


question = 'Is the Lat Pulldown considered a strength training activity, and if so, why?'
answer = rag(question)
print(answer)


'''
# In[57]:


question = 'I want some core exercise to help my back'
answer = rag(question)
print(answer)


# In[19]:


question = 'I want some core calisthenics exercise to help my back'
answer = rag(question)
print(answer)


# ## Retrival Evaluation ##

# In[11]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[13]:


df_question.head(10)


# In[14]:


ground_truth = df_question.to_dict(orient='records')


# In[15]:


ground_truth[0]


# In[16]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[17]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[18]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[19]:


from tqdm.auto import tqdm


# In[20]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# ## Finding the best parameters ##

# In[21]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[27]:


df_validation.head()


# In[22]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score


# In[23]:


gt_val = df_validation.to_dict(orient='records')


# In[24]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[25]:


param_ranges = {
    'exercise_name': (0.0, 3.0),
    'type_of_activity': (0.0, 3.0),
    'type_of_equipment': (0.0, 3.0),
    'body_part': (0.0, 3.0),
    'type': (0.0, 3.0),
    'muscle_groups_activated': (0.0, 3.0),
    'instructions': (0.0, 3.0),
}


# In[26]:


evaluate(gt_val, lambda q: minsearch_search(q['question']))


# In[31]:


def objective(boosting_parameter):
    def search_function(q):
        return minsearch_search(q['question'], boosting_parameter)
    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[32]:


simple_optimize(param_ranges, objective, n_iterations=20 )


# In[38]:


def minsearch_search_improved(query):
    boost = {
        'exercise_name': 2.452138963655172,
        'type_of_activity': 0.0945815257416881,
        'type_of_equipment': 0.8425326046764732,
        'body_part': 2.0368138833148777,
        'type': 1.1566684186139289,
        'muscle_groups_activated': 0.3188239377113703,
        'instructions': 0.5591269022107551}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[39]:


evaluate(ground_truth, lambda q: minsearch_search_improved(q['question']))


# ## RAG Evaluation ##

# In[58]:


prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[59]:


ground_truth = df_question.to_dict(orient='records')


# In[60]:


sample = ground_truth[0]
question = sample['question']
answer_llm = rag(question)


# In[43]:


sample


# In[61]:


question


# In[62]:


answer_llm


# In[64]:


prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )
print(prompt)


# In[65]:


evaluation = llm(prompt)


# In[66]:


print(evaluation)


# In[73]:


import json
evaluations = []
df_sample = df_question.sample(n=100, random_state=1)
sample = df_sample.to_dict(orient='records')

for rec in tqdm(sample):
    question = rec['question']
    answer_llm = rag(question) 
    doc_id = rec['id']  
    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((rec, answer_llm, evaluation))


# In[74]:


#evaluations


# In[72]:


#len(evaluations)


# In[85]:


df_eval = pd.DataFrame(evaluations, columns =['record','answer_llm','evaluation'])
#df_eval.shape


# In[86]:


#df_eval.head()


# In[88]:


df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])
df_eval['Relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['Explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])


# In[90]:


del df_eval['record']
del df_eval['evaluation']


# In[91]:


df_eval.head()


# In[92]:


df_eval['Relevance'].value_counts()


# In[93]:


df_eval['Relevance'].value_counts(normalize = True)


# In[ ]:
'''


