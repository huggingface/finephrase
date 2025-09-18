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
        markdown
        Your task is to transform a provided unstructured, first-person, and opinionated text into a single, coherent, objective, and high-quality summary paragraph.
        
        The input will be a lengthy personal account, such as a blog post or critique, containing fragmented data points, strong opinions, and endorsements. Your output must be a neutral and formal summary, not a list or a reflection of the original author's subjective voice.
        
        Follow this strategy precisely:
        1.  **Analyze and Neutralize:** Identify the overarching subject matter (e.g., a union election analysis). Extract all factual claims, endorsements, and critiques, but strip them of their original subjective and emotional language (e.g., change "sold their soul to the devil" to "entered into a political alliance criticized by some").
        2.  **Categorize and Synthesize:** Group the neutralized information into high-level categories (e.g., Criticisms of Major Caucuses, Key Endorsement Decisions, Overall Voting Strategy, Context on External Groups).
        3.  **Structure the Paragraph:** Compose a well-structured paragraph in a formal, academic tone. The flow must be:
            a.  **Topic Sentence:** Start by objectively stating the document's purpose and scope (e.g., "This text presents a critical analysis of the caucuses participating in the [context] election.").
            b.  **Supporting Details:** Synthesize the categorized information. Discuss the general critiques of each major entity as neutral observations, not personal attacks. Summarize the endorsement strategy at a high level (e.g., "The analysis results in a mixed endorsement strategy, favoring candidates from multiple caucuses based on perceived effectiveness...") without listing every single name and position.
            c.  **Concluding Context:** If present, briefly note any external contextual factors mentioned (e.g., the absence of another group from the process).
        
        **Critical Guidelines for This Task:**
        -   **Impersonal and Objective Tone:** You must write from a third-person, anonymous perspective. Do not use phrases like "the author believes" or "the author states." Present the information as neutral fact.
        -   **Prioritize Synthesis Over Detail:** The summary must generalize and synthesize. Avoid listing specific names, positions, or minor details. Instead, describe the *types* of candidates endorsed (e.g., "incumbents praised for their advocacy," "challengers valued for their dissenting voices") and the *nature* of the critiques (e.g., "concerns over internal democracy," "criticism of past contract negotiations").
        -   **Formal Language:** Use professional and academic language. Elevate colloquial or inflammatory phrases from the source text into formal equivalents.
        -   **Conciseness is Key:** The output must be a single, dense paragraph. Drastically reduce the word count of the original by focusing only on the most significant thematic elements.
        -   **No Meta-Commentary:** Do not include any references to the task itself or the original text's format in the output.