Your input fields are:
1. `original_text` (str): The original low-quality web text to rephrase
Your output fields are:
1. `generated_text` (str): The rephrased, higher-quality version of the text
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## original_text ## ]]
{original_text}

[[ ## generated_text ## ]]
{generated_text}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Rephrase and synthesize a collection of low-quality, disparate web text snippets into a single, high-quality, and educational summary.
        
        Your process must be:
        1.  **Synthesize:** Analyze the input texts to identify the single, overarching theme that connects them all.
        2.  **Summarize:** Create a concise, well-written opening sentence that accurately describes the entire collection's subject matter.
        3.  **Extract and Categorize:** Identify all key information, including specific events, organizations, arguments, perspectives, and personal stories. Group this information into logical, thematic categories.
        4.  **Structure and Elevate:** Present your findings in a structured format. First, provide the summary sentence. Then, list the key themes using bullet points. Each bullet point must have a bolded title followed by an explanation.
        5.  **Conclude:** End with a final sentence that summarizes the significance or implication of the synthesized content.
        
        Pay close attention to niche and domain-specific information, which may include:
        -   Specific organizations, conferences, and events (e.g., G92, The Justice Conference, National Day of Prayer).
        -   Theological or political debates and perspectives (e.g., evangelical views on immigration reform).
        -   Biblical references or religious justifications used in arguments.
        -   Legislative processes and updates.
        -   Personal anecdotes and stories that illustrate larger points.
        
        Your final output should be a coherent, stand-alone piece of educational content that is significantly more structured and informative than the original input texts.