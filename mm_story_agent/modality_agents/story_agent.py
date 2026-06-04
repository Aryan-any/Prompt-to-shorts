import json
from typing import Dict, List
import random
import os

from tqdm import trange, tqdm

from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_en import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system, base_story1,\
    out_story1, story_prompt, scene_prompt, scene_prompt_review_system, story_prompt_review_system
from mm_story_agent.utils.llm_utils import get_llm_config

print("Registering QAOutlineStoryWriter...")

def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() != {"story_title", "story_outline"}:
            return False
        for chapter in outline["story_outline"]:
            if chapter.keys() != {"chapter_title", "chapter_summary"}:
                return False
    except json.decoder.JSONDecodeError:
        return False
    return True


@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        print("Initializing QAOutlineStoryWriter")
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 2)
        
        # Get LLM type from environment or config
        self.llm_type = cfg.get("llm", "gemini")
        
        # Validate LLM type
        if self.llm_type not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def generate_outline(self, params):
        # Configure LLM for asker
        asker_config = {
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system,
                "track_history": False
            }
        }
        asker = init_tool_instance(get_llm_config(asker_config))
        
        # Configure LLM for expert
        expert_config = {
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system,
                "track_history": False
            }
        }
        expert = init_tool_instance(get_llm_config(expert_config))

        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {params}\nDialogue history: \n{dialogue_history}\n",
                temperature=self.temperature
            )
            question = question.strip()
            print(f"\n\n Asker Question:\n {question}\n")
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            answer, success = expert.call(
                f"Story setting: {params}\nQuestion: \n{question}\nAnswer: ",
                temperature=self.temperature
            )
            answer = answer.strip()
            print(f"\n\n Expert Answer:\n {answer}\n")
            dialogue.append(f"Expert: {answer}")
            
        # print("\n".join(dialogue))
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system,
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline
        )

        outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
        print(f"\n\n Writer Outline:\n {outline}\n")
        outline = json.loads(outline)
        # print(outline)
        return outline

    def generate_story_from_outline(self, outline):
        if not isinstance(outline, dict) or "story_outline" not in outline:
            print("Invalid outline format:", outline)
            raise ValueError("Invalid outline format")
        
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": chapter_writer_system,
                "track_history": False
            }
        })
        all_pages = []
        
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            print(f"\nGenerating chapter {idx + 1}: {chapter['chapter_title']}")
            chapter_input = json.dumps(
                {
                    "completed_story": all_pages,
                    "current_chapter": chapter
                },
                ensure_ascii=False,
                indent=2
            )
            print(f"Input to chapter writer:\n{chapter_input}")
            
            chapter_detail, success = chapter_writer.call(
                chapter_input,
                success_check_fn=parse_list,
                temperature=self.temperature
            )
            print(f"Raw chapter response:\n{chapter_detail}")
            
            try_count = 1
            while success is False and try_count < 5:
                print(f"Retry {try_count} for chapter {idx + 1}")
                chapter_detail, success = chapter_writer.call(
                    chapter_input,
                    seed=random.randint(0, 100000),
                    temperature=self.temperature,
                    success_check_fn=parse_list
                )
                print(f"Retry response:\n{chapter_detail}")
                try_count += 1
                
            if not success:
                print(f"Failed to generate chapter {idx + 1} after {try_count} attempts")
                continue
                
            try:
                pages = [page.strip() for page in eval(chapter_detail)]
                print(f"Processed pages:\n{json.dumps(pages, indent=2)}")
                all_pages.extend(pages)
            except Exception as e:
                print(f"Error processing chapter {idx + 1}: {e}")
                continue
                
        return all_pages
    
    def generate_story_from_base_story(self, params):
        story = ""
        review = ""
        language = params["story_lang"]
        for turn in trange(self.max_conv_turns):
            writer = init_tool_instance({
                "tool": self.llm_type,
                "cfg": {
                "system_prompt": story_prompt.format(base_story = params,story_lang= language,base_story1=base_story1, out_story1=out_story1, past_story ="", past_story_criticism = ""),
                "track_history": False
            }
            })
            story, success = writer.call(story_prompt.format(base_story = params,story_lang= language,base_story1=base_story1, out_story1=out_story1, past_story =story, past_story_criticism = review))
            story = story.strip()
            print(f"\n\n story Writer for turn {turn+1}:\n {story}\n")
            story_reviewer = init_tool_instance({
                "tool": self.llm_type,
                "cfg": {
                    "system_prompt": story_prompt_review_system.format(response_story = story, story_lang= language,base_story = params),
                    "track_history": False
                }
            })
            review, success = story_reviewer.call(story_prompt_review_system.format(response_story = story,story_lang= language, base_story = params))
            review = review.strip()
            print(f"\n\n Story Reviewer for turn {turn+1}:\n {review}\n")
            

            
        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            scene_extractor = init_tool_instance({
                "tool": self.llm_type,
                "cfg": {
                    "system_prompt": scene_prompt.format(response_story = story, dialogue_history = dialogue_history),
                    "track_history": False
                }
            })
            scenes, success = scene_extractor.call(scene_prompt.format(response_story = story, dialogue_history = dialogue_history))
            scenes = scenes.strip()
            dialogue.append(f"turn {turn+1}: you: {scenes}")
            print(f"\n\n Scene Extractor for turn {turn+1}:\n {scenes}\n")
            
            scene_reviewer = init_tool_instance({
                "tool": self.llm_type,
                "cfg": {
                    "system_prompt": scene_prompt_review_system.format(response_story = story, response = scenes, dialogue_history = dialogue_history),
                    "track_history": False
                }
            })
            
            review, success = scene_reviewer.call(scene_prompt_review_system.format(response_story = story, response = scenes, dialogue_history = dialogue_history))
            review = review.strip()
            print(f"\n\n Scene Reviewer for turn {turn+1}:\n {review}\n")
            dialogue.append(f"turn {turn+1}: reviewer: {review}")
        
        
        story_with_scenes = json.loads(scenes)
        
        return story_with_scenes


    def call(self, params: Dict) -> List[str]:
        print("Generating story outline...")
        # outline = self.generate_outline(params)
        # print(f"\n\nGenerated outline:\n {json.dumps(outline, indent=2)}\n")
        # generated_story = self.generate_story_from_outline(outline)
        # print(f"\n\nGenerated story:\n {json.dumps(generated_story, indent=2)}\n")
        generated_story = self.generate_story_from_base_story(params)
        print(f"\n\nGenerated story:\n {json.dumps(generated_story, indent=2, ensure_ascii=False)}\n")
        
        return generated_story
