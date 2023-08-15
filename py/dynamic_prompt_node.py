import random
_choice = ["YES", "NO"]
_range = ["Fixed", "Random"]
class DynamicPrompt:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    
                    "variable_prompt": ("STRING", {"forceInput": True} ),
                    "cached": (_choice,{"default": "NO"} ),
                    "number_of_random_tag": (_range,{"default": "Random"} ),
                    "fixed_number_of_random_tag": ("INT", {"default": 1, "min": 1}),

                    },
                    "optional":{
                        "fixed_prompt": ("STRING", {"forceInput": True} ),
                    }
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "prompt_generator"
    CATEGORY = "conditioning"
    OUTPUT_NODE = True

    """
    Generates a prompt based on fixed_prompt and variable_prompt.

    Args:
        fixed_prompt (str): The fixed prompt for generating the prompt.
        variable_prompt (str): The variable prompt for generating the prompt.

    Returns:
        tuple: A tuple containing the generated prompt.
    """
    def prompt_generator(self, variable_prompt,cached,number_of_random_tag,fixed_number_of_random_tag, fixed_prompt = ""):
        prompt = ""
        if fixed_prompt == "undefined":
            fixed_prompt = ""
        #if not an int
        if not str(fixed_prompt):
            fixed_prompt = ""
        if variable_prompt == "undefined":
            variable_prompt = ""
        #if not an int
        if not str(variable_prompt):
            variable_prompt = ""
        try:
            if  variable_prompt.strip() != "":
                add_feature = []
                if len(variable_prompt.split(",")) > 0:
                    variable_prompt_list = variable_prompt.split(",")
                    if number_of_random_tag == "Random":
                        random_range = random.randint(1, len(variable_prompt_list))
                    else:
                        if fixed_number_of_random_tag <= len(variable_prompt_list):
                            random_range = fixed_number_of_random_tag
                        else:
                            random_range = len(variable_prompt_list)

                    for i in range(random_range):
                        len_list = len(variable_prompt_list)-1
                        add_feature.append(variable_prompt_list.pop(random.randint(0, len_list)).strip())
                    # random.choice(variable_prompt.split(","))
                    if fixed_prompt.strip() != "":
                        prompt = f"{fixed_prompt},{ ','.join(add_feature) }"
                    else:
                        prompt = ','.join(add_feature)
                else:
                    print("Warning: Variable prompt is empty.")

            
        except Exception as e:
            
            print(f"Error: something went wrong.\n {e}")
            prompt = fixed_prompt
        return {"ui": {"text": prompt}, "result": (prompt,)}

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DynamicPrompt": DynamicPrompt
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicPrompt": "Dynamic Prompt"
}
