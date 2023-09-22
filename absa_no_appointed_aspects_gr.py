import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
from langchain.chains import LLMChain
from pkg_resources import get_distribution
print(get_distribution('langchain').version)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import gradio as gr
import numpy as np

def str2json(json_string):
        # Split string into lines 
    lines = json_string.split('\n')

    # Initialize empty dictionary
    data = {}

    # Iterate through lines 
    for line in lines:
        # Split line into key-value pair
        try:
            if len(line)>1:
                parts = line.split(': ')
        
                # Extract key and value
                key = parts[0].strip('"')
                value = parts[1]
        
                # Add to dictionary
                data[key] = value
        except:
            continue

    return data

response_schemas = [
    ResponseSchema(name="aspect_term", description="aspect term"),
    ResponseSchema(name="sentiment", description="sentiment, can be positive, neutral or negative")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

'''
template = """This the review from Amazon. The tile of the review is: {title}. The content if the review is: {review}. \
    You are expert in NLP and aspect based sentiment analysis. \
    Please step by step do aspect base sentential analysis with the review and the title and output in json format. \
    Please give out your reply in the json format, every aspect and its sentiment as: \
    aspect term: class(Neutral, positive, or negative.). \
    Automatically find all aspects and their sentiments. Output all of them in complete Json format. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \n{aspect_term}\n{sentiment}"""
'''

template =  """This the review from Amazon. The tile of the review is: {title}. The content of the review is: {review}. \
    You are expert in NLP and aspect based sentiment analysis. \
    Please step by step find all aspects, which are nouns or phrases, and output all aspects in python list, \
    for example: ['aspect1', 'aspect2', 'aspect3']"""
prompt = PromptTemplate(
    template=template,
    input_variables=["title", "review"],
    #partial_variables={"aspect_term": format_instructions, "sentiment":format_instructions}
)

#model = OpenAI(openai_api_key='xxx', temperature=0)
model = OpenAI(temperature=0)
'''
title_ = "The hype is real, I stand corrected"
review_ = "For so long I never bought these because I didn’t want to support the kardashians financially since they promoted it often, however now years later, \
    I tried it and I love it. A family friend who went to cosmetology school, and hair school, \
    recommended these to me after I told her that ever since I cut my hair super short in college I feel like my hair doesn’t grow pst a certain length anymore. \
    I started taking these and my lashes are fuller and my brows are fuller! \
    Best of all my hair is literally noticeably longer than it ever has been other than when I was a little girl. \
    I didn’t want to believe it but it works. I will note that I have a sensitivity to biotin, but most HSN type of vitamins have biotin in them. \
    Because of this I have the tendency to break out when taking anything with it as an incredibly even in the smallest amount. \
    This is something I never was looking to “sacrifice” with other vitamins because they didn’t show much results and I would break out quite a bit, \
    even on my back (I NEVER break out there). My break outs while taking sugar bear were not the worst, but I was getting more than one which I never do. \
    I use a pretty consistent routine that otherwise leaves my skin pretty clear so I feel like I notice everything on my face. Will be repurchasing !"

_input = prompt.format_prompt(title=title_, review=review_)

output = model(_input.to_string())
'''
def absa(title, review):
    _input = prompt.format_prompt(title=title, review=review)
    output = model(_input.to_string())
    return output


import pandas as pd
df = pd.read_csv('./data/ABSA.csv')
df_data = df.loc[:, ['title', 'content']]
df_data = df_data.iloc[:3,:]
df_data['sentiment_analysis'] = np.nan

for idx in range(len(df_data)): ############################ go over row
    tt = df_data['title'].values[idx]
    rev = df_data['content'].values[idx]

    result = absa(tt, rev)
    #aa = str2json(result)
    df_data.loc[idx, 'sentiment_analysis'] = result

df_data.to_csv('SA_without_appointed_aspect.csv')

debug = 1

'''
with gr.Blocks() as demo:
    title_input = gr.Textbox(label="Title")
    review_input = gr.Textbox(label="Review")   
    absa_output = gr.Textbox(label="ABSA output")
    greet_btn = gr.Button("submit")

    greet_btn.click(fn=absa, inputs=[title_input, review_input], outputs=absa_output, api_name="Aspect Based Sentiment Analysis")
   

demo.launch(share=False)
'''
debug = 1

