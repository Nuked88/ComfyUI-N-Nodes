class IntVariable:
   
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": 0

                }),
                
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "check_int"
    CATEGORY = "Variables"

    def check_int(self, value):
        if value == "":
            value = 0
        if value == "undefined":
            value = 0
        #if not an int
        if not int(value):
            value = 0     

        return (value,)

class FloatVariable:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
      
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0,
                    "min": 0

                }),
                
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "check"
    CATEGORY = "Variables"

    def check(self, value):
        if value == "":
            value = 0.0
        if value == "undefined":
            value = 0.0
        #if not an int
        if not float(value):
            value = 0.0

        return (value,)
    
class StringVariable:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", {"multiline": True})}}


    RETURN_TYPES = ("STRING",)
    FUNCTION = "check"
    CATEGORY = "Variables"

    def check(self, value):
        if value == "undefined":
            value = ""
        #if not an int
        if not str(value):
            value = ""
        
        

        return (value,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Integer Variable": IntVariable,
    "Float Variable": FloatVariable,
    "String Variable": StringVariable
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Variables": "Integer Variable",
    "Variables": "Float Variable",
    "Variables": "String Variable"
}
