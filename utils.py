from langchain.document_loaders import TextLoader, YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import streamlit as st

from sklearn.cluster import KMeans

import tiktoken

import numpy as np

from elbow import calculate_inertia, determine_optimal_clusters

import time

import urllib.parse

from concurrent.futures import ThreadPoolExecutor, as_completed


def doc_loader(file_path: str):
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def token_counter(text: str):
    encoding = tiktoken.get_encoding('cl100k_base')
    token_list = encoding.encode(text, disallowed_special=())
    tokens = len(token_list)
    return tokens


def doc_to_text(document):
    text = ''
    for i in document:
        text += i.page_content
    special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|']
    words = text.split()
    filtered_words = [word for word in words if word not in special_tokens]
    text = ' '.join(filtered_words)
    return text

def remove_special_tokens(docs):
    special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|>']
    for doc in docs:
        content = doc.page_content
        for special in special_tokens:
            content = content.replace(special, '')
            doc.page_content = content
    return docs



def embed_docs_openai(docs, api_key):
    docs = remove_special_tokens(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return vectors


def kmeans_clustering(vectors, num_clusters=None):
    if num_clusters is None:
        inertia_values = calculate_inertia(vectors)
        num_clusters = determine_optimal_clusters(inertia_values)
        print(f'Optimal number of clusters: {num_clusters}')

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    return kmeans


def get_closest_vectors(vectors, kmeans):
    closest_indices = []
    for i in range(len(kmeans.cluster_centers_)):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    return selected_indices


def map_vectors_to_docs(indices, docs):
    selected_docs = [docs[i] for i in indices]
    return selected_docs


def create_summarize_chain(prompt_list):
    template = PromptTemplate(template=prompt_list[0], input_variables=([prompt_list[1]]))
    chain = load_summarize_chain(llm=prompt_list[2], chain_type='stuff', prompt=template)
    return chain


def parallelize_summaries(summary_docs, initial_chain, progress_bar, max_workers=4):
    doc_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(initial_chain.run, [doc]): doc.page_content for doc in summary_docs}

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]

            try:
                summary = future.result()

            except Exception as exc:
                print(f'{doc} generated an exception: {exc}')

            else:
                doc_summaries.append(summary)
                num = (len(doc_summaries)) / (len(summary_docs) + 1)
                progress_bar.progress(num)  # Remove this line and all references to it if you are not using Streamlit.
    return doc_summaries





def create_summary_from_docs(summary_docs, initial_chain, final_sum_list, api_key, use_gpt_4):

    progress = st.progress(0)  # Create a progress bar to show the progress of summarization.
    # Remove this line and all references to it if you are not using Streamlit.

    doc_summaries = parallelize_summaries(summary_docs, initial_chain, progress_bar=progress)

    summaries = '\n'.join(doc_summaries)
    count = token_counter(summaries)

    if use_gpt_4:
        max_tokens = 7500 - int(count)
        model = 'gpt-4'

    else:
        max_tokens = 3800 - int(count)
        model = 'gpt-3.5-turbo'

    final_sum_list[2] = ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=max_tokens, model_name=model)
    final_sum_chain = create_summarize_chain(final_sum_list)
    summaries = Document(page_content=summaries)
    final_summary = final_sum_chain.run([summaries])

    progress.progress(1.0)  # Remove this line and all references to it if you are not using Streamlit.
    time.sleep(0.4)  # Remove this line and all references to it if you are not using Streamlit.
    progress.empty()  # Remove this line and all references to it if you are not using Streamlit.

    return final_summary


def split_by_tokens(doc, num_clusters, ratio=5, minimum_tokens=200, maximum_tokens=2000):
    text_doc = doc_to_text(doc)
    tokens = token_counter(text_doc)
    chunks = num_clusters * ratio
    max_tokens = int(tokens / chunks)
    max_tokens = max(minimum_tokens, min(max_tokens, maximum_tokens))
    overlap = int(max_tokens/10)

    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
    split_doc = splitter.create_documents([text_doc])
    return split_doc


def extract_summary_docs(langchain_document, num_clusters, api_key, find_clusters):
    split_document = split_by_tokens(langchain_document, num_clusters)
    vectors = embed_docs_openai(split_document, api_key)

    if find_clusters:
        kmeans = kmeans_clustering(vectors, None)

    else:
        kmeans = kmeans_clustering(vectors, num_clusters)

    indices = get_closest_vectors(vectors, kmeans)
    summary_docs = map_vectors_to_docs(indices, split_document)
    return summary_docs


def doc_to_final_summary(langchain_document, num_clusters, initial_prompt_list, final_prompt_list, api_key, use_gpt_4, find_clusters=False):
    initial_prompt_list = create_summarize_chain(initial_prompt_list)
    summary_docs = extract_summary_docs(langchain_document, num_clusters, api_key, find_clusters)
    output = create_summary_from_docs(summary_docs, initial_prompt_list, final_prompt_list, api_key, use_gpt_4)
    return output


def summary_prompt_creator(prompt, input_var, llm):
    prompt_list = [prompt, input_var, llm]
    return prompt_list







