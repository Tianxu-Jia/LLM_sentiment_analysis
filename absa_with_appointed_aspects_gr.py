from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import gradio as gr

response_schemas = [
    ResponseSchema(name="aspect_term", description="aspect term"),
    ResponseSchema(name="sedntiment", description="sentiment, can be positive, neutral or negative")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """This the review from Amazon customer. The tile of the review is: {title}. The content if the review is: {review}. \
    You are expert in NLP and aspect based sentiment analysis. Please do aspect based sentential analysis with the review and the title in the following aspects:\
        # Effectiveness
        # Price
        # Taste
        # Side effects
        # Quality control
        # Varied results
        # Ingredients. 

    Please give out your reply in the format as: 
    aspect term: class(Neutral, positive, or negative.).
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \n{aspect_term}\n{sentiment}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["title", "review"],
    partial_variables={"aspect_term": format_instructions, "sentiment":format_instructions}
)

#model = OpenAI(openai_api_key='xxx', temperature=0)
model = OpenAI(temperature=0)
def absa(title, review):
    _input = prompt.format_prompt(title=title, review=review)
    output = model(_input.to_string())
    return output

with gr.Blocks() as demo:
    title_input = gr.Textbox(label="Title")
    review_input = gr.Textbox(label="Review")   
    absa_output = gr.Textbox(label="ABSA output")
    greet_btn = gr.Button("submit")

    greet_btn.click(fn=absa, inputs=[title_input, review_input], outputs=absa_output, api_name="Aspect Based Sentiment Analysis")
   

demo.launch(share=True)
