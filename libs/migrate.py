import sys
import os

def migrate_workflow(input_file_path):
    try:
        file_name, file_extension = os.path.splitext(input_file_path)
        
        output_file_path = f"{file_name}_migrated.json"

        pre_list = ('LoadVideo', 'SaveVideo','FrameInterpolator', 'LoadFramesFromFolder','SetMetadataForSaveVideo','GPT Loader Simple','GPTSampler','String Variable','Integer Variable','Float Variable','DynamicPrompt')
        post_list= ('LoadVideo [n-suite]', 'SaveVideo [n-suite]','FrameInterpolator [n-suite]', 'LoadFramesFromFolder [n-suite]','SetMetadataForSaveVideo [n-suite]','GPT Loader Simple [n-suite]','GPT Sampler [n-suite]','String Variable [n-suite]','Integer Variable [n-suite]','Float Variable [n-suite]','DynamicPrompt [n-suite]')
        replacements = list(zip(pre_list, post_list))

        with open(input_file_path, 'r') as input_file:
            content = input_file.read()
              
            # s&r  
            for old, new in replacements:
                content = content.replace(f'"Node name for S&R": "{old}"', f'"Node name for S&R": "{new}"')
            #type
            for old, new in replacements:
                content = content.replace(f'"type": "{old}"', f'"type": "{new}"')

        with open(output_file_path, 'w') as output_file:
            output_file.write(content)
        
        print("Replacement completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) != 2:
        print("Error: Provide the path of the text file to migrate.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)
    
    migrate_workflow(file_path)