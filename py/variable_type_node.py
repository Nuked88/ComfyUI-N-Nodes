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
    CATEGORY = "N-Suite/Variables"

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
    CATEGORY = "N-Suite/Variables"

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
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check"
    CATEGORY = "N-Suite/Variables"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}




    def check(self, string):
        if string == "undefined":
            string = ""
        #if not an int
        if not str(string):
            string = ""
        
        return (string,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Integer Variable [n-suite]": IntVariable,
    "Float Variable [n-suite]": FloatVariable,
    "String Variable [n-suite]": StringVariable
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Integer Variable [n-suite]": "Integer Variable [🅝-🅢🅤🅘🅣🅔]",
    "Float Variable [n-suite]": "Float Variable [🅝-🅢🅤🅘🅣🅔]",
    "String Variable [n-suite]": "String Variable [🅝-🅢🅤🅘🅣🅔]"
}
