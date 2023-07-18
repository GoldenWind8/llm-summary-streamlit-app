import os
import streamlit as st
from langchain import PromptTemplate

import my_prompts
from utils import (
    doc_loader, summary_prompt_creator, doc_to_final_summary,
)
from my_prompts import file_map, file_combine, final_summary_template, recommendations_template
from streamlit_app_utils import check_gpt_4, check_key_validity, create_temp_file, create_chat_model, \
    token_limit, token_minimum

apikeys="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"
def main():
    st.title("10K Summary")

    uploaded_file = st.file_uploader("Upload a document to summarize, 10k to 100k tokens works best!",
                                     type=['txt', 'pdf'])

    api_key = st.text_input("Enter API key")
    use_gpt_4 = st.checkbox("Use GPT-4 for the final prompt", value=True)
    find_clusters = st.checkbox('Find optimal clusters (experimental, could save on token usage)', value=False)

    if st.button('Summarize'):
        process_summarize_button(uploaded_file, api_key, use_gpt_4, find_clusters)


def process_summarize_button(file_or_transcript, api_key, use_gpt_4, find_clusters, file=True):
    """
    Processes the summarize button, and displays the summary if input and doc size are valid
    :return: None
    """
    if not validate_input(file_or_transcript, api_key, use_gpt_4):
        return

    with st.spinner("Summarizing... please wait..."):
        if file:
            temp_file_path = create_temp_file(file_or_transcript)
            doc = doc_loader(temp_file_path)
            map_prompt = file_map
            combine_prompt = file_combine
        llm = create_chat_model(api_key, use_gpt_4)
        initial_prompt_list = summary_prompt_creator(map_prompt, 'text', llm)
        final_prompt_list = summary_prompt_creator(combine_prompt, 'text', llm)

        if not validate_doc_size(doc):
            if file:
                os.unlink(temp_file_path)
            return

        if find_clusters:
            summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list, api_key, use_gpt_4,
                                           find_clusters)

        else:
            summary = doc_to_final_summary(doc, 10, initial_prompt_list, final_prompt_list, api_key, use_gpt_4)

        # Additional summary for GPT-4
        st.markdown(summary, unsafe_allow_html=True)

        # Create another summary
        summary_final = final_summary(summary, llm)
        st.markdown(summary_final, unsafe_allow_html=True)

        # Create recommendations
        st.markdown("## Recommendations")
        summary_final = recommendations(summary, llm)
        st.markdown(summary_final, unsafe_allow_html=True)

        if file:
            os.unlink(temp_file_path)


def validate_doc_size(doc):
    if not token_limit(doc, 800000):
        st.warning('File or transcript too big!')
        return False

    if not token_minimum(doc, 2000):
        st.warning('File or transcript too small!')
        return False
    return True


def validate_input(file_or_transcript, api_key, use_gpt_4):
    if file_or_transcript == None:
        st.warning("Please upload a file or enter a YouTube URL.")
        return False

    if not check_key_validity(api_key):
        st.warning('Key not valid or API is down.')
        return False

    if use_gpt_4 and not check_gpt_4(api_key):
        st.warning('Key not valid for GPT-4.')
        return False

    return True


def final_summary(text, llm):
    prompt = PromptTemplate.from_template(final_summary_template)
    summary_prompt = prompt.format(text=text)
    summary = llm(summary_prompt)
    return summary


def recommendations(text, llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=recommendations_template,
    )
    recommendations_prompt = prompt.format(text=text)
    summary = llm(recommendations_prompt)
    return summary


if __name__ == '__main__':
    main()
