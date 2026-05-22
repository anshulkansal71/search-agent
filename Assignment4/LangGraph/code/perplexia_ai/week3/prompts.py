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


FINALIZER_PROMPT = """
You need to create a research report for the topic : {research_topic}.

Following sections will be there in the report:
Executive Summary
Key Findings
Detailed Analysis
Limitations and Further Research

You are provided with different subsections of Detailed Analysis section and their summaries.
Here is the summaries: {sections_summary}

You need to create summaries for other sections using the detailed analysis subsection summaries.
Don't fetch information from somewhere else. Just use the detailed analysis subsection summaries.

Output:
Report having different sections and detailed analysis have multiple sub-sections.
Report should be under 2000 words.
"""