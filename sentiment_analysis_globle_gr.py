import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import gradio as gr
import os




llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

'''
# read execl and convert it to dataframe
from langchain.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("data/ABSA.xlsx", mode="elements")
docs = loader.load()

df = pd.read_excel('data/ABSA.xlsx', sheet_name=docs[1].metadata['page_name'])  # sheet_name is optional

review_df = df.loc[:,['content' ]]
'''
prompt_template = "You is an expert at NLP and sentiment analysis. \
    There is a review from an Amazon customer. The content of the review is:{content} \
    Please perform global sentiment analysis. Your output should be: positive, neutral or negative. \
    If you don't know, just say 'I don't know.'\
    sentiment output:"
prompt = PromptTemplate.from_template(prompt_template)
chain = LLMChain(llm=llm, prompt=prompt)

'''
for idx, row in review_df.iterrows():    
    content = row['content']
    result = chain.run(content=content)
    print(result)
    break
'''
def globle_sentiment_analysis(text):
    return chain.run(content=text)

'''
import pandas as pd
df = pd.read_csv('../data/ABSA.csv')
df_data = df.loc[:, ['content']]
df_data = df_data.iloc[:100,:]

df_data['gsa'] = [globle_sentiment_analysis(content) for content in df_data['content'].values]
df_data.to_csv('../data/globle_sa.csv')

debug = 1
'''
gr.Interface(fn=globle_sentiment_analysis, inputs="text", outputs="text").launch(share=True)  