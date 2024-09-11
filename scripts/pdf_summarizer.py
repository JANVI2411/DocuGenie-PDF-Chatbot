"""# Summarizer

https://python.langchain.com/v0.1/docs/use_cases/summarization/
"""

from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'facebook/bart-large-cnn'# go for a smaller model if you dont have the VRAM
fb_tokenizer = AutoTokenizer.from_pretrained(model_id)
fb_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

fb_pipe = pipeline(
    "text2text-generation",
    model=fb_model,
    tokenizer=fb_tokenizer,
    max_length=100
)

summary_llm = HuggingFacePipeline(pipeline=fb_pipe)


"""### Method2: Stuff chain"""

print("Original Text: \n")
for split in all_splits[:5]:
  print(split.page_content)

"""#### stuff implementation: 1"""

from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(summary_llm, chain_type="stuff")

summary = chain.run(all_splits[:5])

len(summary), summary

"""#### stuff implementation: 2"""

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain import LLMChain

simple_template ="""
                  Write a concise summary of the following:
                  {text}
                  Concise Summary: """

prompt = PromptTemplate(template=simple_template, input_variables=["text"])

simple_llmchain = LLMChain(llm = summary_llm, prompt = prompt)

stuff_chain = StuffDocumentsChain(llm_chain=simple_llmchain,document_variable_name="text")

summary = stuff_chain.run(all_splits[:5])

len(summary), summary

"""### Method3: Map-Reduce Method"""

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

# Map-chain
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""

map_prompt = PromptTemplate.from_template(map_template)

map_chain = LLMChain(llm=summary_llm, prompt=map_prompt)


# Reduce-chain
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_chain = LLMChain(llm=summary_llm, prompt=reduce_prompt)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000,
)

#Final Map-Reduce chain
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

summary = map_reduce_chain.run(all_splits[:5])

print(summary)

"""### Method4: Refine"""

prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in English"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=summary_llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": all_splits[:5]}, return_only_outputs=True)

result["output_text"]

print("\n\n".join(result["intermediate_steps"]))

