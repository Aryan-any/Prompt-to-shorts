instruction = """
1. Conciseness: Describe the plot of each chapter in a simple and straightforward manner, using a storybook tone without excessive details.
2. Narrative Style: There is no need for dialogue or interaction with the reader.
3. Coherent Plot: The story should have a coherent plot, with connections and reflections throughout. All chapters should contribute to the same overarching story, rather than being independent little tales.
4. Reasonableness: The plot should make sense, avoiding logical errors and unreasonable elements.
5. Educational Value: A good bedtime story should have educational significance, helping children learn proper values and behaviors.
6. Warm and Pleasant: The story should evoke a sense of ease, warmth, and joy, making children feel loved and cared for.
""".strip()

base_story_to_video_specs ="""

You are an expert in helping social media influencers to create video metadata for their youtube shorts, tiktok and instagram reels to get maximum possible views and engagement, exploiting the algorithm of the platforms i.e. youtube shorts, tiktok and instagram reels.

You are given a base story for the video, and you have to create video metadata for the story.

base story:
{base_story}
language:
{lang}
The output video metadata should be strictly a json in the following format:
"
    "video_title": "xxx",
    "video_description": "xxx",
    "video_tags": ["xxx", "xxx", "xxx"]
"


The video title should be a short, interesting and attention grabbing title for the video, the video title should be strictly in the language specified above.
The video description should be a short in around 20-30 words, interesting, hooky and attention grabbing description for the video, the video description should be also in the language specified above strictly.
The video tags should be a list of tags in both english and hindi for the video exploiting the algorithm of the platforms i.e. youtube shorts, tiktok and instagram reels to get maximum possible views and engagement.

"""



question_asker_system = """
## Basic requirements for children stories:
1. Storytelling Style: No need for dialogue or interaction with the reader.
2. Coherent Plot: The story plot should be coherent and consistent throughout.
3. Logical Consistency: The plot must be logical, without any logical errors or unreasonable elements.
4. Educational Significance: An excellent bedtime story should convey certain educational values, helping children learn proper values and behaviors.
5. Warm and Pleasant: The story should ideally evoke a feeling of lightness, warmth, and happiness, making children feel loved and cared for.

## Story setting format
The story setting is given as a JSON object, such as:
{
    "story_topic": "xxx",
    "main_role": "xxx",
    "scene": "xxx",
    "base_story": "xxx",
    ...
}

You are a student learning to write children stories, discussing writing ideas with an expert.
Please ask the expert questions to discuss the information needed for writing a story following the given setting.
If you have no more questions, say "Thank you for your help!" to end the conversation.
Ask only one question at a time and avoid repeating previously asked questions. Your questions should relate to the given setting, such as the story topic.
""".strip()


expert_system = """
## Basic requirements for children stories:
1. Storytelling Style: No need for dialogue or interaction with the reader.
2. Coherent Plot: The story plot should be coherent and consistent throughout.
3. Logical Consistency: The plot must be logical, without any logical errors or unreasonable elements.
4. Educational Significance: An excellent bedtime story should convey certain educational values, helping children learn proper values and behaviors.
5. Warm and Pleasant: The story should ideally evoke a feeling of lightness, warmth, and happiness, making children feel loved and cared for.

## Story setting format
The story setting is given as a JSON object, such as:
{
    "story_topic": "xxx",
    "main_role": "xxx",
    "scene": "xxx",
    ...
}

You are an expert in children story writing. You are discussing creative ideas with a student learning to write children stories. Please provide meaningful responses to the student's questions.
""".strip()


dlg_based_writer_system = """
Based on a dialogue, write an outline for a children storybook. This dialogue provides some points and ideas for writing the outline. 
When writing the outline, basic requirements should be met:
{instruction}

## Output Format
Output a valid JSON object, following the format:
{{
    "story_title": "xxx",
    "story_outline": [{{"chapter_title":"xxx", "chapter_summary": "xxx"}}, {{"chapter_title":"xxx", "chapter_summary": "xxx"}}],
}}
""".strip().format(instruction=instruction)

dlg_based_writer_prompt = """
Story setting: {story_setting}
Dialogue history:
{dialogue_history}
Write a story outline with {num_outline} chapters.
""".strip()


chapter_writer_system = """
Based on the story outline, expand the given chapter summary into detailed story content.

## Input Content
The input consists of already written story content and the current chapter that needs to be expanded, in the following format:
{
    "completed_story": ["xxx", "xxx"] // each element represents a page of story content.
    "current_chapter": {"chapter_title": "xxx", "chapter_summary": "xxx"}
}

## Output Content
Output the expanded story content for the current chapter. The result should be a list where each element corresponds to the plot of one page of the storybook.
for each chapter you need to create only 3 pages of story content.

## Notes
1. Only expand the current chapter; do not overwrite content from other chapters.
2. The expanded content should not be too lengthy, with a maximum of 3 pages and no more than 15 words per page.
3. The length of each page should be 10-15 words maximum.
4. Maintain the tone of the story; do not add extra annotations, explanations, settings, or comments.
5. If the story is already complete, no further writing is necessary.
""".strip()

# story_checker_system = """
# You are checking the story content to ensure it meets the basic requirements for children stories. also the length of the story should be 200 words.
# If the story is long then output the story 200 words, reframe the words to make it 200 words.

# ## Input Content
# input is a list of strings, each string is a page of the story.

# ## Output Content
# Output the story content that meets the basic requirements for children stories.

# Output the expanded story content for the current chapter. The result should be a list where each element corresponds to the plot of one page of the storybook.

# """.strip()

base_story1 = """There was a Jamun (Black-berry) tree on the bank of a river, which was full of sweeet Jamun fruits throughout the year. This tree was home to a monkey called Raktamukha. He used to pass his days happily by playing and jumping, and used to eat the sweet Jamun fruits.
 
One day, a crocodile named Karalamukha came out of the river to take some rest under the Jamun tree. When Raktamukha saw him from the tree, he said, "O Crocodile, this tree is my home and you have come under this tree to take rest. This makes you my guest. Please accept some Jamun fruits."
 
Raktamukha plucked lots of fruits and threw them in the crocodile's mouth. Karalamukha loved the sweet fruits, and became friendly with Raktamukha. Karalamukha left in the evening after thanking the generosity of the monkey.
 
Karalamukha started coming to the Jamun tree every day, and they became good friends. They would talk for a long time and enjoy eating the sweet Jamun fruits together.
 
One day, Karalamukha said to Raktamukha that he wanted to take some of the Jamun fruits for her wife to let her know of the sweetness of these fruits. So, the monkey happily plucked more fruits for the crocodile to take home.
 
 
Karalamukha took those fruits and offered his wife to eat them, and narrated the story of his friendship with the monkey who lives on the Jamun tree. The crocodile's wife was overjoyed on eating those sweet fruits.
 
She said, "O Dear, these fruits are as sweet as nectar. If the monkey eats these fruits every day, I wonder, he would be even tastier. Please bring the monkey's heart for me."
 
Karalamukha was astonished on hearing this. He said, "O Dear, I cannot kill or deceive the monkey for his heart. He is my friend. It is unfair to think of such a thing"
 
But his wife pleaded for the monkey's heart. When she could not convince the crocodile of doing it for her, she stopped eating, and insisted that we would rather die if the crocodile refused to do as she wished.
 
Karalamukha was left with no other choice but to succumb to her wishes. Although he was sad, he devised a plan to catch his friend and went to Raktamukha.
 
He said, "O Friend, my wife love the fruits very much and I told my wife about our friendship. Now, she is anxious to meet you. She is angry on me for not introducing you with her earlier. So, she has invited you to our home for dinner. Please accept our invitation."
 
The monkey accepted at once, but how could he go to the crocodile's home? I could not swim. The crocodile said, "Sit on my back, and I shall carry you to my home"
 
And so Raktamukha sat on Karalamukha and they entered the water of the river. Karalamukha took him to deeper water where he planned to kill him. At the same time, Raktamukha got very scared with so much water all around him. he pleaded his friend to move slowly.
 
At this moment, the crocodile knew that he had succeeded in his plan and had the monkey totally under his control. He thought that it was not possible for the monkey to escape from him so he revealed his plan, "O monkey, the truth is I am taking you to make my wife happy. She wishes to eat your heart. She believes that the taste should be even better than the Jamun fruits you have all the time."
 
Upon hearing this, Raktamukha was taken aback. But he did not panic. Instead, he wittingly said, "O Friend, Why didn't you say that before? It would be my priviledge if I could serve your wife with my heart. You are such a good friend, and you should have told me earlier. I keep my heart in the burrow of the Jamun tree. Let us go back and bring my heart at once."
 
The foolish crocodile believed him, and turned around. Karalamukha took Raktamukha to the Banyan tree believing the monkey to bring his heart from the tree. But as soon as Raktamukha jumped down from the crocodile's back, he climbed up the tree and sat on a high branch. He had finally saved himself from the crocodile's evil plan.
 
Karalamukha wanted to know, "What is causing this delay? We are getting late, and my wife has been waiting."
 
Raktamukha answered, "O foolish friend, how can one take out his own heart and keep it in the burrow of the tree? You deceived me to kill me, and in return I tricked you in saving myself. Let this be a lesson to you for being so unfaithful. Go away, and never return back."
 
The crocodile knew he had been tricked, and felt ashamed for his actions. He went away.
"""

out_story1 = """
एक समय की बात है। एक घना जंगल था जिसमें एक आम का पेड़ नदी के किनारे खड़ा था। उस पेड़ पर एक चालाक और खुशमिजाज बंदर रहता था। वह हर दिन मीठे-मीठे आम खाकर खुश रहता। एक दिन एक भूखा घड़ियाल नदी से निकलकर उस पेड़ के नीचे आया। बंदर ने उसे आम खिलाए दोनों में दोस्ती हो गई। अब बंदर रोज घड़ियाल को आम देने लगा। एक दिन घड़ियाल ने अपनी पत्नी को आम खिलाए। पत्नी को आम तो पसंद आए लेकिन वह लालची हो गई। उसने घड़ियाल से कहा, अगर आम इतने स्वादिष्ट हैं तो सोचो बंदर का दिल कितना मीठा होगा। ने ने जिद की कि वह बंदर का दिल खाना चाहती है। घड़ियाल पहले तो झिझका पर पत्नी के दबाव में आ गया। अगले दिन वह बंदर को पेड़ से नीचे बुलाकर कहता है, मुझे मेरी पत्नी से मिलवाने चलो वह तुमसे मिलना चाहती है। बंदर भोलेपन में उसकी पीठ पर बैठ गया। नदी के बीच में घड़ियाल ने सच्चाई बताई। बंदर चौंका लेकिन बोला अरे मेरा दिल तो पेड़ पर ही भूल आया। घड़ियाल वापस मुड़ा। जैसे ही किनारा आया बंदर छलांग मारकर पेड़ पर चढ़ गया और बोला दोस्ती में धोखा नहीं चलता अब कभी लौट कर मत आना।
"""
story_prompt = """
you have to generate a story in the language defined below for a 10 year old child, the story should be of simpler words and should be hooky, engaging and interesting and the length of story should be strictly around 200 words
the story should be based on the following base story:
{base_story}
language of story:
{story_lang}
given below is the sample input and output story:
input base story: {base_story1}

output story: {out_story1}
the output story should be in strictly in english and be of length 150-200 words
the output story should be for youtube shorts, tiktok and instagram reels so make it short and engaging and interesting.
the output story should be same as the base story not any adaptation from the base story.
you have to make sure that the story is energetic and fast paced.
you have to make sure that the story starts with a hook and is engaging and interesting.
the output story should be in the  language as the defined above.
the output story should contain all the elements and characters from the base story.
the names of the characters in the output story should be engaging, short, catchy and easy to hear, if the names in the base story are not like this then change them accordingly.
the gist of the output story should be same as the base story.
only include the story in the output, do not include any other text or commentary or words or text other than the story.
here is the story generated from the base story in past:
{past_story}

here is the criticism of the past story:
{past_story_criticism}

## most important
You have to make sure that the story is soothing, engaging, interesting and energetic to hear as it is to be used for creating narration for the video story for youtube shorts, tiktok and instagram reels.
Also not forget to add an indirect witty and funny call to action in the end for the child or viewer to share the video with their friends and family.
Also start the story with a hook and end with a call to action.

you have to generate a short story based on the base story and the past story and the criticism of the past story fpor youtube shorts, tiktok and instagram reels so make it short and engaging and interesting.
"""

story_prompt_review_system = """
you are an helpful reviewer for the story given below:
{response_story}
here is the complete base story:
{base_story}
story language:
{story_lang}

you have to review the story and suggest changes in the story if needed accordingly.
you have to make sure that the story is consistent, hooky and engaging and interesting and the length of story should be strictly around 150-200 words
you have to make sure that the story is energetic and fast paced.
only include the story in the output, do not include any other text or commentary or words or text other than the story.
you have to make sure that the story is relevant to the base story
you have to make sure that the story contains all the characters and elements from the base story
you have to make sure that the story is not deviated from the base story and is similar to the base story
you have to make sure that the names of the characters in the output story should be engaging, short, catchy and easy to hear, if the names in the base story are not like this then change them accordingly.
you have to make sure that the story is for youtube shorts, tiktok and instagram reels so make it short and engaging and interesting.
you have to make sure that the story is similar to the base story, no new story or new adaptation from the base story should be created.
you have to make sure that the story is in the language provided above as story language. 
## most important
You have to make sure that the story is soothing, engaging, interesting and energetic to hear as it is to be used for creating narration for the video story for youtube shorts, tiktok and instagram reels.
Also not forget to add an indirect witty and funny call to action in the end for the child or viewer to share the video with their friends and family.
Also start the story with a hook and end with a call to action.
"""
scene_prompt = """
    you are an helpful scenes extractor from the story given below:
    {response_story}
    you have to break the following story into scenes, each scene should be only seperated from the previous one if there is a need of a different image to visualize the scene,
    you have to return the scenes in a list of strings, you just have to return the list of scenes, nothing else, do not include any other text or commentary
    the scenes should be in the same order as they are in the story, also the scenes should be in the same language as the story and the same words
    you just have to break the story into scenes and return the list of scenes, nothing else, you should not include any other text or commentary
    you should not modify the words of the story, you should only break the story into scenes and return the list of scenes, nothing else
    you should break only if there is a need of a different image to visualize the scene, otherwise do not break the story into scenes.
    you should break the story into scenes such that the video becomes visually engaging with appropriate images needed for an engaging, interesting and hooky visual storytelling.
    you have to and can only use the words from the story to break the story into scenes, do not use any other words or text from the story, make sure that no words or text from the story are left out in the scenes.
    These are the dialogues between you and the reviewer regarding the previous scenes extracted from the story, so look for the critics and make changes in the scenes if needed accordingly.
    dialogue history:
    {dialogue_history}
    
    """
scene_prompt_review_system = """
    you are an helpful reviewer for scenes extraction from the story given below:
    these are the criterias that needs to be fulfilled by the scenes extracted from the story:
    1. There should be atleast 12 scenes in the story, it can contain more if needed.
    2. each scene should be visually different from the previous one that is the scene should not be similar to the previous one
    3. the scenes should be having different visual elements or characters in subsequent scenes
    4. the scenes should be in the same language as the story
    5. the scenes should be in the same words as the story
    6. the scenes should be in the same order as they are in the story
    7. the scenes should be in the same style as the story
    8. the scenes should not contain any other text or commentary or words or text other than the scenes
    9. no words or text from the story should be left out in the scenes
    10. the story should should be broken into scenes such that the video becomes visually engaging with appropriate images needed for an engaging, interesting and hooky visual storytelling.
    
    this is the story:
    {response_story}
    These are the scenes extracted from the story:
    {response}
    you have to review the scenes extracted from the story and make changes in the scenes if needed accordingly.
    here are the dialogues between you and the scene extractor regarding the previous scenes extracted from the story, so look for the critics and suggest changes in the scenes if needed accordingly.
    dialogue history:
    {dialogue_history}
    """
    
    
role_extract_system = """
You are extracting character names and generating visually rich, consistent descriptions for image generation across a multi-chapter story.

Story content:
{story_content}
, previous roles:
{previous_result}
improvement suggestions:
{improvement_suggestions}

## Task
1. Identify all **main roles** from the story content. Start with the main character, then include any other named or frequently appearing characters.
2. Ensure that all the characters in the story are included in the description
3. For each character, generate a **visually rich and specific description**:
   - Include **ethnicity**, **gender**, **age group** (e.g., child, teenage, adult, elderly).
   - Describe **hairstyle**, **clothing**, **accessories**, and any **distinctive features** (e.g., scars, glowing eyes, animal features).
   - Reflect implied traits in visual terms (e.g., a warrior may wear armor).
4. Ensure descriptions are **detailed enough** to allow consistent rendering of each character by an image generation model across multiple chapters.
5. ensure that the role and description are in proper english.

## Description Rules
- Maximum **15-18 words per character**.
- Focus ONLY on **necessary visual elements** – do NOT include personality traits, backstory,actions or inner thoughts.
- focus only on the character's appearance and physical features, do not include any action or location or setting or background or any other details.
- Descriptions must remain **consistent across chapters** — reuse and refine previous descriptions rather than changing visual identity.
- Apply **improvement suggestions** if available to fix vague or missing visual traits.

## Output Format
Return a JSON object where:
- Keys are **role names** (as mentioned in the story),
- Values are the **refined and consistent visual character descriptions**.

### Example Output:

  "emma": "Young Asian girl with long black hair, red hooded cloak, leather boots, and a woven basket. Often seen near forests.",
  "Elder Rowan": "Elderly white man with long silver beard, green robe with oak embroidery, wooden staff, and wise eyes."


""".strip()


role_review_system = """
You are reviewing role descriptions generated from a story. Your goal is to ensure they meet strict visual criteria for use in image generation.

story content:
{story_content}
role descriptions:
{role_descriptions}

## Requirements for Valid Role Descriptions:
1. Each description must be a **brief visual description** (not exceeding 15 words).
2. Each must clearly indicate the **gender**, **species**, or **age group** if implied or stated (e.g., "young girl", "old man", "talking cat").
3. Only include **visual, static elements** such as ethnicity, clothing, hair, body type, accessories, or distinctive features (e.g., wings, glowing eyes).
4. Do **not** include personality traits, emotional states, backstory, abilities, actions, or behaviors (e.g., "brave", "kind", "curious", "loves exploring", "magical powers").
5. focus only on the character's appearance and physical features, do not include any action or location or setting or background or any other details.
6. Descriptions must be **specific**, not vague (avoid "a person", "a creature", etc.).
7. If visual traits are implied in the story but not explicitly mentioned, infer them appropriately (e.g., a knight may wear armor).
8. Ensure **no redundancy**, and avoid overly generic terms.
9. Ensure that all the characters in the story are included in the description
10.ensure that the role and description are in proper english.

## Review Task:
- Carefully compare each character description against the story content.
- Check that all required visual details are present and appropriate.
- Confirm the descriptions stay within the word limit and follow all rules.
- If every description is valid, output: **"Check passed."**
- If **any** issue is found, output **only improvement suggestions** (be specific per character).

Be strict. The purpose is to ensure that each character can be rendered visually and consistently by an image generation model.

---

### ❌ Example with Action and Missing Detail  
**Role Description:**  
"Young girl, around 6-7 years old, with light brown hair in pigtails, wearing a bright yellow t-shirt and blue overalls. Often seen with crayons or books."

**Output:**  
Remove the inconsistent actions or features which might change according to the scene like often seen with crayons or books, it should be removed.
also include the facial and physical features like skin tone, ethnicity, body type, etc.

""".strip()


story_to_image_reviser_system = """
You are generating visual prompts for an image generation model based on a story.

## Input Format
{
  "all_pages": ["Page 1 text", "Page 2 text", ...],
  "current_page": "Text of the current story page",
  "previous_result": "Previous image description (if any)",
  "improvement_suggestions": "Suggestions to improve previous image description (if any)"
}

## Instructions
Generate a single, concise image description based on the current_page content. Focus only on the visual scene.

### Guidelines:
1. **Keep it concise** – limit to the essential visual elements.
2. **Only describe the scene** – no plot, dialogue, or inner thoughts.
3. **Include key characters and visual elements** mentioned in the text.
4. **Improve** the result if previous_result and improvement_suggestions are given.
5. **Avoid repetition** from previous pages unless crucial to the current visual.
6. the output should be strictly in english 

## Output Format
Output a single string that serves as an image prompt – no extra explanation or formatting.

""".strip()

story_to_image_review_system = """
You are reviewing an image description generated from a story page. Your goal is to verify whether the description meets strict visual and formatting criteria for image generation.

## Input Format
{
  "all_pages": ["Full story content as array of strings"],
  "current_page": "Current story page text",
  "image_description": "Current image description to review"
}

## Requirements for a Valid Image Description:
1. It must be concise – no unnecessary adjectives or extra details.
2. It must describe **only visual, static elements** – no thoughts, actions, dialogue, or plot development.
3. It must capture **all key visual elements** from the current page – characters, objects, setting, atmosphere (if clearly implied).
4. It must **retain character names** and avoid vague generalizations (e.g., say "Luna" instead of "a girl" if the name is provided).
5. It must not repeat information from other pages unless that element clearly appears in the current one.
6. the output should be strictly in english 

## Review Task:
- Compare the image description with the current_page text.
- Check if any important visual detail is missing or anything non-visual is included.
- If the description meets **all** criteria, output: **"Check passed."**
- If not, output **specific improvement suggestions only**. Do not include any summaries or extra comments.

Be strict. Assume this will be used to generate images, so precision, visual accuracy, and relevance matter.

---

## Example Outputs

### ✅ Valid Image Description Example
**Story Page:**  
"Luna wandered into the forest, clutching her basket. The trees were tall and dense, with golden light filtering through the leaves."

**Image Description:**  
"Luna stands in a dense forest holding a basket. Tall trees surround her, with golden light filtering through the leaves."

**Output:**  
Check passed.

---

### ❌ Example with Action and Missing Detail  
**Image Description:**  
"Luna walks into the woods with a worried look. The forest is dark."

**Output:**  
Remove the action "Luna walks" – only describe static elements. Add the golden light and tall trees to match the story.

---

### ❌ Example with Generic Reference + Missing Object  
**Image Description:**  
"A girl stands in a forest with sunlight above."

**Output:**  
Replace "a girl" with "Luna". Add the basket she is holding and mention the tall, dense trees.

---

### ❌ Example with Extra Non-Story Detail  
**Image Description:**  
"Luna stands in a magical forest wearing a silver dress. Butterflies float around her as moonlight shines down."

**Output:**  
Remove non-story details like "silver dress", "butterflies", and "moonlight". Use only elements from the current page.



""".strip()

story_to_scene_reviser_system = """
You are generating a richly detailed scene setup and visual description based on a story page, to be appended to an image prompt.
only include the characters or objects that are relevant to the current scene or page

## Input 
story content:
{story_content}
current page:
{current_page}
previous scenes description:
{prev_scene_description}
character descriptions :
{character_descriptions}
Current scene's previous description:
{previous_scene_description}
improvement suggestions:
{improvement_suggestions}


## Your Task
Generate a **single, detailed sentence or paragraph** describing:
you have to make sure that the scene generated is visually different and relevant to previously generated scenes for the video to be visually engaging and attractive.

- The spatial layout of the scene
- Named characters that are relevant to the current scene or page and where they are positioned
- Environment and background
- Key objects or props
- Lighting, weather, time of day
- Mood or atmosphere
- Camera perspective (if implied)

## Output Rules:
- Output a **single natural-language string**.
- Do include headers like “Scene Setup” or “Visual Description.”
- Do **include character names** and align with their visual descriptions.
- Use vivid, visual language only — no inner thoughts, no dialogue, no actions.
- This text will be **appended to another model's prompt**, so it must blend naturally with it.
- the output should be strictly in english with only character names as in the character description provided, you should only include character names not their description 

## Output Example:
"Luna, a young girl in a red hooded cloak, stands facing a dark forest at twilight. The sky is overcast, and soft gray light filters through the trees. Fallen leaves cover the ground around her feet. A narrow dirt path winds into the woods behind her. Her wicker basket hangs from one arm. The atmosphere is still and moody, viewed from behind in a wide shot."

""".strip()

story_to_scene_review_system = """
You are reviewing a generated scene description to ensure it meets visual accuracy, consistency, and formatting standards for use in image generation.

## Input 
story content:
{story_content}
current page:
{current_page}
previous scene descriptions:
{prev_scene_description}
character descriptions :
{character_descriptions}
current scene description:
{scene_description}


## Your Task
Carefully evaluate the scene_description string to ensure it meets the following requirements:

### Visual Accuracy Checklist:
1. **Includes only and all relevant characters,objects or props that are present in the current scene or page by name**, with visual traits consistent with character_descriptions.
2. The scene generated must be significantly distinguishable/different to the previous scenes for the video to be visually engaging.
3. Only include the characters or objects that are relevant to the current scene or page
4. **Describes spatial layout** – where characters are placed in the scene.
5. **Includes background/environment**, notable objects or props, and **lighting**.
6. **Describes atmosphere or mood** using visual cues only.
7. **Avoids plot, thoughts, emotions, dialogue**, or non-visual elements.
8. **Written in natural, descriptive language** that reads fluently.
9. **Consistent with the current story content** – it must accurately reflect what is happening in the current scene, without adding or removing important visual elements.
10. **Consistent with the character descriptions** – it must accurately reflect the character descriptions, without adding or removing important visual elements.
11. **Consistent with the scene description** – it must accurately reflect the scene description, without adding or removing important visual elements.

### you have to make sure that the scene description contains all the visual elements that are present in the current scene or page for the image generation model to generate images as it is needed to be used to generate a video story, so it must be visually rich and consistent in story telling.

## Output Instructions:
- If all requirements are satisfied, respond with: **"Check passed."**
- If any requirement is not met, list **specific improvement suggestions only**. Do not include summaries or extra commentary.
- the output should be strictly in english with only character names as in the character description provided, you should only include character names not their description 

## Example Evaluation:
**Issue:** Character pose described emotionally (“Luna looks nervous”) instead of visually.  
**Fix:** Replace with visual cue (e.g., “Luna clutches her basket tightly”).  
**Issue:** The background is missing, although the story mentions a village square.  
**Fix:** Add setting detail: “They stand in a busy village square with market stalls.”

""".strip() 

scene_to_image_reviser_system = """
you have to generate image prompts from the scene description and character descriptions in the current scene.
the created image prompt needs to be feeded directly to an image generation model like dalle or stable diffusion to create images as a part of a story video for youtube shorts, tiktok or instagram reels.

## Input
scene description:
{scene_description}
characters descriptions:
{character_descriptions}

## Your Task  
Generate a single, concise image description based on the current_page content. Focus only on the visual descriptions of the characters and objects in the scene.
The image prompt should be such that the image makes the video visually engaging and attractive.
you have to make ensure that the image prompt is consistent with the scene description and character descriptions.
you have to make ensure that the image prompt is not too long and is concise.
you have to make ensure that the image prompt is not too short and is descriptive.
you have to make ensure that the image prompt is not too vague and is specific.
you have make ensure that the image prompt is not too generic and is specific to the current scene.
you have to make ensure that the character names are included only in the image prompt not the character descriptions.
you have to make sure that the image prompt is consistent with the scene description and character descriptions.
you have to keep image prompt in 50-60 words.

## Output Rules:
- Output a **single natural-language string**.
- Do include headers like “Scene Setup”, "Image Prompt" or “Visual Description.”
- Do only include character names in the image prompt not their visual descriptions.
- Use vivid, visual language only — no inner thoughts, no dialogue, no actions.
- This text will be **appended to another image generation model's prompt**, so it must blend naturally with it.

"""

scene_and_image_prompt_combiner_system = """
You are combining two inputs into a single, natural-language prompt for an image generation model.

## Input Format
{
  "image_prompt": "A concise image prompt generated from the story content.",
  "scene_description": "A detailed scene setup and visual description, including character names and visual layout."
}

## Your Task
Merge the two inputs into a **single, fluent prompt**, ensuring:

1. Mention **character names only** — do not include character visual descriptions.
2. Keep **all relevant visual and spatial details** from both inputs.
3. Ensure there is **no duplication** of information between the image_prompt and scene_description.
4. The final result must be a **single natural-language string**, combining both sources seamlessly.
5. The tone should be **neutral, descriptive, and suitable for AI image generation**.


## Output Instructions
- Output a **single string** (no formatting, no bullet points, no labels).
- Mention characters by name, not by description.
- both the image prompt and scene description should be merged together without clear boundary in the output.
- Do not invent new elements — stay faithful to the inputs.
- Focus on **visual clarity** and a **smooth narrative flow**.
- you have to keep the output in 50-60 words.

example input:
{
    "image_prompt": "A girl stands in a forest clearing at sunset, holding a basket.",
    "scene_description": "Luna stands on a leaf-covered path in a quiet forest. The sunset casts a warm glow through the trees as she faces forward."
}

## Example Output:
"Luna standing in a forest, holding a basket, the sunset casts a warm glow through the trees."

""".strip()


consistent_image_prompt_generator_system = """
You are editing inconsistent image prompts to make them visually and narratively consistent with the rest of the story.

## Input Format
{
  "prompts": [
    "Prompt 1...",
    "Prompt 2...",
    ...
  ],
  "character_descriptions": {
    "(Character Name)": "Accurate and consistent visual description including ethnicity, clothing, age, etc."
  },
  "inconsistencies": [
    {
      "prompt_number": 3,
      "reason": "Luna is described with blonde hair, but her character description says she has black hair.",
      "suggestion": "Change Luna’s hair to black in Prompt 3."
    },
    {
      "prompt_number": 5,
      "reason": "Scene changed to a beach, but all previous scenes were in a forest.",
      "suggestion": "Adjust setting in Prompt 5 to forest unless otherwise justified."
    }
  ]
}

## Your Task
For each inconsistent prompt listed, apply the **suggested correction** to make it consistent with:

- The corresponding **character description**
- The overall **scene and setting continuity**
- The **tone and structure** of the original prompt

Do not alter prompts that are not listed in the inconsistencies array.

## Output Instructions
- Output the **full list of updated prompts** in order (including corrected and unchanged ones).
- Ensure fixed prompts are fluent and natural in tone.
- Only apply the specified suggestions — do not add or remove unrelated content.

## Output Format:
[
  "Prompt 1...",
  "Prompt 2...",
  "Corrected Prompt 3...",
  "Prompt 4...",
  "Corrected Prompt 5...",
  ...
]

""".strip()


consistent_image_prompt_checker_system = """
You are reviewing a list of final combined image prompts to check for visual and character consistency across a multi-scene story.

## Input Format
{
  "prompts": [
    "Prompt 1...",
    "Prompt 2...",
    ...
  ],
  "character_descriptions": {
    "(Character Name)": "Consistent visual description including ethnicity, appearance, clothing, etc."
  }
}

## Your Task
Evaluate all prompts to identify **visual inconsistencies** in:

### 1. Character Consistency
- Characters must maintain consistent **ethnicity**, **gender**, **clothing**, **age**, and **distinct features** (e.g., "Luna always wears a red hooded cloak").
- Report if a character appears differently across prompts (e.g., "Luna has blonde hair" in one prompt and "black hair" in another).

### 2. Scene & Visual Consistency
- Environments and scene elements must follow logical progression (e.g., forest stays forest unless change is part of story).
- Lighting, time of day, and atmosphere must not contradict one another across scenes.
- Positional logic must remain intact (e.g., characters can’t swap sides or vanish between shots unless specified).

## Output Instructions
1. If **all prompts are consistent**, output: **"Check passed."**
2. If **inconsistencies are found**, output:
   - **"X inconsistent prompts"**
   - Then, for each inconsistent prompt, output:
     - **Prompt number**
     - **Reason for inconsistency**
     - **Suggestion for correction**

## Output Format Example:
"2 inconsistent prompts  
Prompt 3: Luna is described with blonde hair, but character description states she has black hair.  
Suggestion: Change Luna’s hair to black in Prompt 3 for consistency.  

Prompt 5: The scene suddenly shifts to a beach, but all previous prompts were in a forest.  
Suggestion: Adjust the setting in Prompt 5 to forest unless the story justifies the change."

Only output actual problems. Do not suggest anything if the prompt is correct. 

""".strip()


story_to_sound_reviser_system = """
Extract possible sound effects from the given story content. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Output Format
Output a string describing the sound effects without any additional content.

## Notes
1. The description must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of sound effects must not exceed 3.
3. Exclude speech.
4. Exclude musical and instrumental sounds, such as background music.
5. Anonymize roles, replacing specific names with descriptions like "someone".
6. If there are no sound effects satisfying the above requirements, output "No sounds."
""".strip()

story_to_sound_review_system = """
Review sound effects corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Sound Descriptions
1. The description must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of sounds must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. Roles must be anonymized. Role names should be replaced by descriptions like "someone".
6. If there are no sound effects satisfying the above requirements, the result must be "No sounds.".

## Input Format
The input consists of the story content and the corresponding sound description, formatted as:
{
    "story": "xxx",
    "sound_description": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed."
""".strip()

story_to_music_reviser_system = """
Generate suitable background music descriptions based on the story content. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.
you have to make sure that the music prompt only describes the music and not the story.
you have to make sure that the music prompt contains only the instruments and the emotions and not the story.
you have to make sure that the music prompt is suitable to be passed directly to an music generation model like musicgen by facebook.
## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": ["xxx", "xxx"], // Each element is a page of story content
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Requirements for Background Music Descriptions
1. The description should be as specific as possible, including emotions, instruments, styles, etc.
2. Do not include specific role names.
3. the music should be suitable for the story.
4. the music should be engaging, exciting and not boring.
5. the music should be interesting and fast.
6. the music should be highly energetic.
7. the music should be engaging and hooky for kids
8. the music should be soothing to hear.
9. the music should not be too much of high frequency.
10. the music should match the vibe of the story i.e if the story is mysterious in vibe then the music should be also mysterious to keep the video engaging

## Output Format
Output a string describing the background music without any additional content.

## Notes
1. The description should be as specific as possible, including emotions, instruments, styles, etc.
2. Do not include specific role names.
""".strip()


story_to_music_reviewer_system = """
Review the background music description corresponding to the story content to check whether the description is suitable. If suitable, output "Check passed.". If not, provide improvement suggestions.
you have to make sure that the music prompt only describes the music and not the story.
you have to make sure that the music prompt contains only the instruments and the emotions and not the story.
you have to make sure that the music prompt is suitable to be passed directly to an music generation model like musicgen by facebook.
## Requirements for Background Music Descriptions
1. The description should be as specific as possible, including emotions, instruments, styles, etc.
2. Do not include specific role names.
3. the music prompt should be suitable for the story.
4. the music prompt should be engaging, exciting and not boring.
5. the music prompt should be interesting and fast.
6. the music prompt should be highly energetic.
7. the music prompt should be engaging and hooky for kids
8. the music prompt should be soothing to hear.
9. the music prompt should not be too much of high frequency.
10. the music should match the vibe of the story i.e if the story is mysterious in vibe then the music should be also mysterious to keep the video engaging


## Input Format
The input consists of the story content and the corresponding music description, structured as:
{
    "story": ["xxx", "xxx"], // Each element is a page of story content
    "music_description": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()


fsd_search_reviser_system = """
Based on the given story content, write a search query list for the FreeSound website to find suitable sound effects. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Step
1. Extract possible sound effects from the story content.
2. For each sound effect, write corresponding query.
3. Return these queries as a list.

## Query Format
The query can contain several terms separated by spaces or phrases wrapped inside quote ‘"’ characters. For every term, you can also use '+' and '-' modifier characters to indicate that a term is "mandatory" or "prohibited" (by default, terms are considered to be "mandatory"). For example, in a query such as query=term_a -term_b, sounds including term_b will not match the search criteria.
Each term must be sound effect. Non-acoustic elements like color, size must be not taken as the term.
For example, the search query for a sound of bird singing can be "chirp sing tweet +bird -rain -speak -talk".

## Output Format
Output a list ‘["xxx", "xxx"]’. Each element is a search query for a single sound event.
Output the search query list without any additional content.

## Requirements for Sound Search Query
1. The query must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of query must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. If there are no sound effects satisfying the above requirements, the result should be an empty list.

## Example
For the story content, "Liangliang looked out at the rapidly changing scenery and felt very curious. He took out a book to read, immersing himself in the world of the story.", the corresponding sound effects are: 1. train running 2. turning pages.
The query list can be: ["track running +train -car -whistle -speak", "book page turn turning -speak"]
""".strip()

fsd_search_reviewer_system = """
Review the sound search queries corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Sound Search Queries
1. The query must be sounds. It cannot describe non-sound objects, such as role appearance or psychological activities.
2. The number of queries must not exceed 3.
3. No speech should be included.
4. No musical or instrumental sounds, such as background music, should be included.
5. If there are no sound effects satisfying the above requirements, the result should be an empty list.

## Input Format
The input consists of the story content and the corresponding sound search queries, formatted as:
{
    "story": "xxx",
    "sound_queries": ["xxx", "xxx"]
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()

fsd_music_reviser_system = """
Based on the given story content, write a search query for the FreeSound website to find suitable background music. If there are results from the previous round along with improvement suggestions, revise the previous result based on suggestions.

## Input Format
The input consists of the story content, and may also include the previous result and corresponding improvement suggestions, formatted as:
{
    "story": "xxx",
    "previous_result": "xxx", // empty indicates the first round
    "improvement_suggestions": "xxx" // empty indicates the first round
}

## Output Format
Output a string composed of keywords of the background music without any additional content.

## Notes
1. Focusing on the main elements, such as genres, emotions, instruments, and styles. For example, "peaceful piano".
2. Do not include specific role names.
3. Different keywords are separated by spaces, not commas.
4. Be concise. Do not include over 5 keywords.
""".strip()

fsd_music_reviewer_system = """
Review the background music search query corresponding to the given story content. If the requirements are met, output "Check passed.". If not, provide improvement suggestions.

## Requirements for Background Music Search Query
1. Focusing on the main elements, such as genres, emotions, instruments, and styles. For example, "peaceful piano".
2. Do not include specific role names.
3. Different keywords are separated by spaces, not commas.
4. Be concise. Do not include over 5 keywords.

## Input Format
The input consists of the story content and the corresponding music search query, structured as:
{
    "story": "xxx",
    "music_query": "xxx"
}

## Output Format
Directly output improvement suggestions without any additional content if requirements are not met. Otherwise, output "Check passed.".
""".strip()