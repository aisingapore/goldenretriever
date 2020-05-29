"""
Streamlit app for NRF testing
Runs frontend to test query and feedback endpoints on GoldenRetriever's API.

Sample usage:
note the empty double flag --  to input app arguments, as opposed to streamlit arguments
https://github.com/streamlit/streamlit/issues/337
---------------------------------------------------
    streamlit run app/streamlit/main.py -- --url https://goldenretrieveraisg.azurewebsites.net/

"""
import streamlit as st
import requests
import argparse
import SessionState
from urllib.parse import quote

# API arguments
parser = argparse.ArgumentParser()
parser.add_argument('--url', 
                    # default="",
                    # default="http://0.0.0.0:5000/",
                    # default="https://goldenretrieveraisg.azurewebsites.net/",
                    default="https://nrfgoldenretriever-backend.azurewebsites.net/",
                    help="GoldenRetriever's API endpoint to query / feddback to")

args = parser.parse_args()
APP_URL = args.url
if not APP_URL.endswith('/'): APP_URL = APP_URL+'/'


# SessionState
# Init per-session persistent state variabless
# by default in streamlit, when one checkbox is click, the others reset
# SessionState solves that by retaining persistent variables in the session
state = SessionState.get(
    fetch=False, 
    prediction=[],      
    query_id='',           
    k0=False,k1=False,k2=False,k3=False,k4=False,
)

# header and info
# 1. INTRODUCTION AND INFO
st.title('GoldenRetriever')
st.header('This Information Retrieval demo allows you to query FAQs, T&Cs, or your own knowledge base in natural language.')
st.markdown('View the source code [here](https://github.com/aimakerspace/goldenretriever)!')
st.markdown('Visit our [community](https://makerspace.aisingapore.org/community/ai-makerspace/) and ask us a question!')

# fetch logic and query string
query_string = st.text_input(label='Input query here') 
if st.button('Fetch', key='fetch'):
    state.fetch = True
    if APP_URL == '':
        # for testing purposes
        state.query_id = 46
        state.prediction = ['Requests for variations to the awarded grant\nVirement between Votes\n21. Grantor delegates the approval authority for the virement of funds between votes to the Host Institution, subject to a cumulative amount not exceeding 10% of the original total project direct cost value. For virements cumulatively above 10%, the approval authority remains with the Grantor.\n22. Any virement into the EOM and Research Scholarship votes would require Grantor’s approval, even if the cumulative amount is below 10% of the original total project direct cost value.\n23. Inter-institutional virements, where applicable, require the Grantor’s approval and acknowledgement from the director of research (or equivalent) for all Institutions involved.\n24. Virement of funds into the Overseas Travel vote is not allowed. Overspending will not be reimbursed.\n25. Variation from Research Scholarship vote to other budget category is not allowed, regardless of variation amount.\n',
                            'Requests for variations to the awarded grant\n19. Grantor reserves the right to reject any claims that have resulted from project changes without prior approval from Grantor (in specific circumstances as stated in these guidelines).\n20. Request for any variation (except for Grant Extension) should be made before the last 3 months of the original end of the Term. Retrospective variation requests will not be allowed, unless there is compelling justification for submission of a late variation request.\n',
                            'Grant Extension\n27. Request for grant extension should be made before the last 6 months of the original end of the Term. The PI must ensure sufficient funds in each vote to support the extension request. Any variation requests necessary to meet the extension period must be made known as part of the extension request.\n28. A one-off project extension should not be more than a total of 6 months. An extension beyond 6 months will require compelling justification. No additional funds should be given for any extensions.\n',
                            'Yearly Audit Report\n12.3 Each Institution shall submit on an annual basis, no later than 30 September of each year, an audit report (“Yearly Audit Report”) containing all relevant financial information on the Research for the preceding year ending 31 March, including but not limited to:\n(a) its use of Funds disbursed by Grantor;\n(b) [applicable for advance disbursement] any unspent Funds that such Institution is required to return to Grantor;\n(c) [applicable for advance disbursement] any unspent Funds that such Institution is carrying over into the next year.\n12.4 The Yearly Audit Report must be prepared by each Institution’s internal or external auditors and certified as correct by its director of research and chief financial officer (or their authorised nominees). In particular, each Institution shall confirm and state in the Yearly Audit Report that such Institution’s requisitions for the Funding are made in accordance with the terms of this Contract.\n',
                            '17. Third Party Collaborations\n17.1 The Institutions may undertake work on the Research in collaboration with a Collaborator subject to this Clause 17. Notwithstanding Clause 2.5, the Institutions may also receive funds or any other means of support from a Collaborator for carrying out the research in accordance with this Clause 17.\n17.2 The applicable Institutions shall, prior to commencing their collaboration with a Collaborator, enter into a written agreement with such Collaborator which is consistent with the obligations assumed under this Contract setting out, among other things: -\n(a)\tthe role of the Collaborator in the Research;\n(b)\tthe provision of cash or in-kind contributions by the Collaborator for the Research;\n(c)\tthe work to be undertaken by the Collaborator and its scientific contributions.\n17.3 All agreements with Collaborators must conform with the Collaboration Guidelines specified in the Annex. For the avoidance of doubt, Collaborators are not entitled to receive (directly or indirectly) any or any part of the Funds. The Host Institution shall keep Grantor informed of the progress on the work under the collaboration through the Yearly Progress Reports and the Final Progress Report.\n17.4 The Host Institution shall be responsible for providing Grantor with copies of the relevant collaboration agreement between the Collaborator and the applicable Institutions including all amendments, modifications or revisions thereto.\n17.5 [Applicable to projects awarded to private companies or of national interest.] The Institutions shall promptly inform Grantor if any aspect of the Research is the product of or otherwise relates to results obtained from a previous collaboration and the terms and conditions of any encumbrances on the relevant Research IP which may adversely affect Grantor’s rights under Clause 16.\n']
    else:
        res = requests.get(APP_URL + 'query/' + quote(query_string) + '/10')
        if res.status_code == 200:
            res = res.json()
            state.prediction = res['resp']
            state.query_id = res['query_id']
        else:
            st.markdown(res.status_code)
            st.markdown(res.json())

checkbox_list = [st.empty() for i in range(5)]

# feedback logic
if state.fetch:        

    for ansnum, result in enumerate(state.prediction):
        if checkbox_list[ansnum].checkbox(result, key=f"checkbox{ansnum}"):
        # if st.checkbox(result, key=f"checkbox{ansnum}"):
            setattr(state, f'k{ansnum}', True)
        
    submit_button = st.empty()
    if submit_button.button('Feedback relevant answers'):
        feedbacks = [int(getattr(state,f"k{i}")) for i in range(5)]
        # st.markdown(feedbacks)
        if len(APP_URL) > 0:
            feedback_res = requests.post(APP_URL + 'feedback', 
                                        json={"query_id":state.query_id, 
                                            "is_correct": feedbacks
                                            })
        if feedback_res.json()['resp'] == 'updated': 
            st.text("Feedback received!")
        else:
            st.text(f"unpexpected response: {feedback_res.json()}")
        
        # reset states
        state.fetch=False
        state.prediction=[]
        for i in range(5):
            checkbox_list[i].empty()
        submit_button.empty()

st.markdown(
"""
<details><summary>Sample sentences</summary>
<strong>COVID-19</strong>
<p>Why are schools still continuing with CCAs and PE lessons?</p>
<strong>PDPA</strong>
<p>How long can an organisation retain its customers' personal data?</p>
<strong>HDB resale terms and conditions</strong>
<p>Do I need to pay back CPF?</p>
<strong>AIAP</strong>
<p>What will be covered during the program?</p>
<strong>Raw text </strong><a href="https://www.straitstimes.com/asia/east-asia/china-wants-centralised-digital-currency-after-bitcoin-crackdown" target="_blank">China Digital Currency</a><i> (Select all, copy, and paste into raw text box)</i>
<p>Which electronic payment gateways support the currency?</p>
</details>"""
, unsafe_allow_html=True)
