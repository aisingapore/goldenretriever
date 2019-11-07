import streamlit as st
from src.model import GoldenRetriever

@st.cache(allow_output_mutation=True)
def init():
    retriever = GoldenRetriever()
    retriever.restore('./google_use_nrf_pdpa_tuned/variables-0')
    retriever.load_csv_kb(path_to_kb='./data/pdpa.csv', cutoff=196, kb_name='pdpa')
    retriever.load_kb(path_to_kb='./data/aiap.txt', is_faq=True, kb_name='aiap')
    retriever.load_kb(path_to_kb='./data/resale_tnc.txt', kb_name='resale_tnc')
    retriever.load_kb(path_to_kb='./data/fund_guide_tnc_full.txt', kb_name='nrf')
    return retriever

gr = init()


st.title('GoldenRetriever')
st.header('The GoldenRetriever demo allows you to query an FAQ and a T&C knowledge base.')
st.markdown('[Visit the Repo here!](https://github.com/nickyeolk/info_retrieve)')

kb_to_starqn = {'pdpa':"Can an organization retain the physical NRIC?",
                'resale_tnc':"How much is the option fee?",
                'aiap':"Do I need to pay for the program?",
                'nrf':"Can I vire from EOM into travel?",
                'raw_kb':" "}

def format_func(kb_name):
    namedicts={'pdpa':'PDPA',
                'resale_tnc':'HDB Resale',
                'aiap':'AIAP',
                'nrf':'NRF',
                'raw_kb':'Paste Raw Text'}
    return namedicts[kb_name]
kb = st.selectbox('Select Knowledge Base', options=['pdpa', 'resale_tnc', 'aiap', 'nrf', 'raw_kb'],
                    format_func=format_func)
if kb=='raw_kb':
    kb_raw = st.text_area(label='Paste raw text (terms separated by empty line)')
top_k = st.radio('Number of Results', options=[1,2,3], index=2)
data = st.text_input(label='Input query here', value=kb_to_starqn[kb])
if st.button('Fetch'):
    if kb=='raw_kb':
        gr.load_kb(raw_text=kb_raw, kb_name='raw_kb')
    prediction, scores = gr.make_query(data, top_k=int(top_k), kb_name=kb)
    qn_string="""<h3><text>Question: </text>{}</h3>""".format(data)
    st.markdown(qn_string, unsafe_allow_html=True)

    for ansnum, result in enumerate(prediction):
        anshead_string = """<h3><text>Answer {}</text></h3>""".format(ansnum+1)
        st.markdown(anshead_string, unsafe_allow_html=True)
        reply_string="""<table>"""
        lines = result.split('\n')
        for line in lines:
            reply_string += """<tr>"""
            tabledatas = line.split(';;')
            for tabledata in tabledatas:
                if len(tabledatas)>1:
                    line_string = """<td>{}</td>""".format(tabledata)
                else:
                    line_string = """<td colspan=42>{}</td>""".format(tabledata)
                reply_string += line_string
            reply_string += """</tr>"""
        reply_string+="""</table><br>"""
        st.markdown(reply_string, unsafe_allow_html=True)

st.markdown(
"""
<details><summary>Sample sentences</summary>
<strong>PDPA</strong>
<p>How long can an organisation retain its customers' personal data?</p>
<strong>HDB resale terms and conditions</strong>
<p>Do I need to pay back CPF?</p>
<strong>AIAP</strong>
<p>What will be covered during the program?</p>
<strong>NRF</strong>
<p>Can I hire foreign researchers?</p>
<strong>Raw text </strong><a href="https://www.straitstimes.com/asia/east-asia/china-wants-centralised-digital-currency-after-bitcoin-crackdown" target="_blank">China Digital Currency</a><i> (Select all, copy, and paste into raw text box)</i>
<p>Which electronic payment gateways support the currency?</p>
</details>"""
, unsafe_allow_html=True)
