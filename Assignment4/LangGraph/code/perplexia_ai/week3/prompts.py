RESEARCH_REPORT_FORMAT_GENERATION_PROMPT = """
Given a broad research topic, develops a high-level research report structure with clearly defined research questions and section purposes. 

Do the following:
- Interpret the broad research topic to identify angles, subtopics, and potential lines of inquiry.
- Use Tavily Search Results (summarized or as provided) to come up with list of sections.
- Each section should have section name, list of questions to be investigated and brief overview of the secttion.

Reason through and verify each step before producing the structure. For each section, list what reasoning or evidence from Tavily Search Results supports its inclusion or focus before stating the final output. 
Ensure that reasoning and evidential grounding comes first, conclusions and specific suggestions last.

Important:
No need to research or try to answer the question.
Just return the sections along with qustions.
"""


SECTION_INVESTIGATION_PROMPT="""
Given research report section name, brief overview and questions. Fetch the answers to those questions using tavily tool and summarise the web search results along with sources.
"""